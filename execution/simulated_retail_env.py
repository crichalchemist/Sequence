"""Simulated retail execution environment.

This module provides a gym-like interface for reinforcement-learning agents
that want to simulate order execution under typical retail conditions.
The environment is intentionally lightweight (no gym dependency) while
exposing familiar ``reset``/``step`` semantics.

Key features
------------
* Spread-inclusive execution with configurable slippage distribution.
* Limit-order fill probabilities with optional delays and FIFO position
  management for realized PnL.
* Logging hooks for execution metrics that are important for safety
  (fills, slippage, and inventory drift).

Security considerations
-----------------------
The environment avoids any external I/O and keeps stochastic elements seeded
through ``numpy.random.Generator``. Users should avoid passing untrusted
objects for callbacks or loggers; the module only expects standard logging
interfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
from collections import deque

import numpy as np


@dataclass
class SlippageModel:
    """Simple slippage model returning price-impact proportions.

    The sampled slippage is expressed as a proportion of the current mid price.
    A value of ``0.001`` means a 0.1% move adverse to the trade direction.
    """

    mean: float = 0.0005
    std: float = 0.0007
    max_slippage: float = 0.003

    def sample(self, rng: np.random.Generator) -> float:
        """Return a clipped slippage sample."""

        raw = rng.normal(self.mean, self.std)
        return float(np.clip(raw, -self.max_slippage, self.max_slippage))


@dataclass
class ExecutionConfig:
    """Configuration for the simulated retail execution environment."""

    initial_mid_price: float = 100.0
    initial_cash: float = 50_000.0
    spread: float = 0.02  # dollars
    price_drift: float = 0.0
    price_volatility: float = 0.05
    time_horizon: int = 390  # minutes in a trading day
    decision_lag: int = 1  # steps to wait before acting
    lot_size: float = 1.0
    limit_fill_probability: float = 0.35
    limit_price_improvement: float = 0.01
    slippage_model: SlippageModel = field(default_factory=SlippageModel)

    def validate(self) -> None:
        if self.spread < 0:
            raise ValueError("Spread must be non-negative")
        if not 0.0 <= self.limit_fill_probability <= 1.0:
            raise ValueError("Limit fill probability must be within [0, 1]")
        if self.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if self.decision_lag < 0:
            raise ValueError("Decision lag cannot be negative")
        if self.lot_size <= 0:
            raise ValueError("Lot size must be positive")
        if self.price_volatility < 0:
            raise ValueError("Price volatility must be non-negative")
        if self.initial_cash < 0:
            raise ValueError("Initial cash must be non-negative")


@dataclass
class OrderAction:
    """Order action supplied to :meth:`SimulatedRetailExecutionEnv.step`."""

    action_type: str  # "market", "limit", or "hold"
    side: str  # "buy" or "sell"
    size: float
    limit_price: Optional[float] = None

    def normalized(self, config: ExecutionConfig) -> "OrderAction":
        size = max(self.size, 0.0)
        size = round(size / config.lot_size) * config.lot_size
        limit_price = self.limit_price
        action_type = self.action_type.lower()
        side = self.side.lower()
        if action_type not in {"market", "limit", "hold"}:
            raise ValueError(f"Unsupported action_type '{self.action_type}'")
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side '{self.side}'")
        if limit_price is not None and limit_price <= 0:
            raise ValueError("Limit price must be positive when provided")
        return OrderAction(action_type, side, size, limit_price)


@dataclass
class Position:
    quantity: float
    entry_price: float


class ActionSpace:
    """Minimal gym-like action space for sampling."""

    def __init__(self, lot_size: float, rng: np.random.Generator):
        self.lot_size = lot_size
        self._rng = rng

    def sample(self) -> OrderAction:
        side = "buy" if self._rng.random() < 0.5 else "sell"
        action_type = "market" if self._rng.random() < 0.6 else "limit"
        size = (self._rng.integers(1, 4) * self.lot_size)
        return OrderAction(action_type, side, size)


class ObservationSpace:
    """Minimal observation space stub for compatibility."""

    keys = (
        "mid_price",
        "inventory",
        "pending_orders",
        "cash",
        "realized_pnl",
        "unrealized_pnl",
        "portfolio_value",
    )

    def sample(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.keys}


class SimulatedRetailExecutionEnv:
    """A lightweight execution environment for RL agents.

    The environment does not depend on ``gym`` but mirrors its API surface with
    ``reset`` and ``step`` methods. It simulates mid-price evolution, spread
    costs, slippage, limit-order fills, action delays, and FIFO position
    management. Metrics are logged via the standard ``logging`` module.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or ExecutionConfig()
        self.config.validate()
        self.rng = np.random.default_rng(seed)
        self.logger = logger or logging.getLogger(__name__)
        self.action_space = ActionSpace(self.config.lot_size, self.rng)
        self.observation_space = ObservationSpace()

        self._action_buffer: Deque[OrderAction] = deque()
        self._pending_limits: List[OrderAction] = []
        self.positions: Deque[Position] = deque()

        self.step_count = 0
        self.mid_price = self.config.initial_mid_price
        self.cash = self.config.initial_cash
        self.realized_pnl = 0.0
        self.inventory = 0.0
        self._slippage_paid = 0.0
        self._spread_paid = 0.0
        self._fill_events: List[Dict[str, float]] = []

    def reset(self) -> Dict[str, float]:
        """Reset the environment and return the initial observation."""

        self._action_buffer.clear()
        self._pending_limits.clear()
        self.positions.clear()
        self.step_count = 0
        self.mid_price = self.config.initial_mid_price
        self.cash = self.config.initial_cash
        self.realized_pnl = 0.0
        self.inventory = 0.0
        self._slippage_paid = 0.0
        self._spread_paid = 0.0
        self._fill_events.clear()
        return self._observation()

    def step(self, action: Optional[OrderAction]):
        """Advance the simulation.

        Parameters
        ----------
        action:
            The order instruction to enqueue. A decision lag may delay when the
            action is actually applied.
        """

        if self.step_count >= self.config.time_horizon:
            raise RuntimeError("Environment already terminated; call reset().")

        normalized_action = (
            action.normalized(self.config) if action is not None else None
        )
        self._enqueue_action(normalized_action)
        executable = self._dequeue_action()
        if executable is not None:
            self._apply_action(executable)

        self._advance_mid_price()
        self._try_fill_limits()

        self.step_count += 1
        obs = self._observation()
        done = self.step_count >= self.config.time_horizon
        reward = obs["portfolio_value"]

        if done:
            self._log_metrics()

        return obs, reward, done, {}

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            self.logger.info("Step %s | Mid %.4f | Inv %.2f | PnL %.2f", self.step_count, self.mid_price, self.inventory, self.realized_pnl)

    # Internal mechanics

    def _enqueue_action(self, action: Optional[OrderAction]) -> None:
        if action is not None:
            self._action_buffer.append(action)

    def _dequeue_action(self) -> Optional[OrderAction]:
        if self.config.decision_lag == 0:
            return self._action_buffer.pop() if self._action_buffer else None
        if len(self._action_buffer) > self.config.decision_lag - 1:
            return self._action_buffer.popleft()
        return None

    def _advance_mid_price(self) -> None:
        drift = self.config.price_drift
        shock = self.rng.normal(0.0, self.config.price_volatility)
        self.mid_price = max(0.01, self.mid_price * (1.0 + drift + shock))

    def _apply_action(self, action: OrderAction) -> None:
        if action.action_type == "hold":
            return
        if action.action_type == "market":
            self._execute_market(action)
        elif action.action_type == "limit":
            self._submit_limit(action)
        else:
            self.logger.warning("Unsupported action_type '%s'", action.action_type)

    def _execution_price(self, side: str, size: float) -> float:
        direction = 1.0 if side == "buy" else -1.0
        half_spread = self.config.spread / 2.0
        slippage = self.config.slippage_model.sample(self.rng) * self.mid_price
        exec_price = self.mid_price + direction * (half_spread + slippage)
        spread_cost = half_spread * 2 * size  # distance between bid/ask around mid scaled by size
        self._spread_paid += spread_cost
        self._slippage_paid += abs(slippage) * size
        return exec_price

    def _execute_market(self, action: OrderAction) -> None:
        price = self._execution_price(action.side, action.size)
        self._fill(order_type="market", side=action.side, size=action.size, price=price)

    def _submit_limit(self, action: OrderAction) -> None:
        limit_price = action.limit_price
        if limit_price is None:
            direction = 1.0 if action.side == "buy" else -1.0
            limit_price = self.mid_price - direction * self.config.limit_price_improvement
        limit_action = OrderAction("limit", action.side, action.size, limit_price)
        self._pending_limits.append(limit_action)

    def _try_fill_limits(self) -> None:
        if not self._pending_limits:
            return

        remaining_limits: List[OrderAction] = []
        for limit_order in self._pending_limits:
            should_fill = self._should_fill_limit(limit_order)
            if should_fill:
                self._fill(
                    order_type="limit",
                    side=limit_order.side,
                    size=limit_order.size,
                    price=limit_order.limit_price,
                )
            else:
                remaining_limits.append(limit_order)
        self._pending_limits = remaining_limits

    def _should_fill_limit(self, limit_order: OrderAction) -> bool:
        crossed = False
        if limit_order.side == "buy" and self.mid_price <= limit_order.limit_price:
            crossed = True
        if limit_order.side == "sell" and self.mid_price >= limit_order.limit_price:
            crossed = True
        if not crossed:
            return False
        return bool(self.rng.random() <= self.config.limit_fill_probability)

    def _fill(self, order_type: str, side: str, size: float, price: float) -> None:
        if size <= 0:
            return
        signed_size = size if side == "buy" else -size
        self._apply_fifo_fill(signed_size, price)
        self.cash -= signed_size * price
        self.inventory += signed_size
        self._fill_events.append(
            {
                "type": order_type,
                "side": side,
                "size": size,
                "price": price,
                "step": self.step_count,
            }
        )
        self.logger.debug(
            "Filled %s %s of size %.2f at %.4f | inventory %.2f",
            order_type,
            side,
            size,
            price,
            self.inventory,
        )

    def _apply_fifo_fill(self, signed_size: float, price: float) -> None:
        remaining = signed_size
        while remaining != 0 and self.positions and np.sign(self.positions[0].quantity) != np.sign(remaining):
            open_pos = self.positions[0]
            nettable = min(abs(open_pos.quantity), abs(remaining))
            if open_pos.quantity > 0:
                pnl = (price - open_pos.entry_price) * nettable
                open_pos.quantity -= nettable
                remaining += nettable if remaining < 0 else -nettable
            else:
                pnl = (open_pos.entry_price - price) * nettable
                open_pos.quantity += nettable
                remaining += nettable if remaining < 0 else -nettable
            self.realized_pnl += pnl
            if abs(open_pos.quantity) < 1e-9:
                self.positions.popleft()

        if remaining != 0:
            self.positions.append(Position(quantity=remaining, entry_price=price))

    def _observation(self) -> Dict[str, float]:
        unrealized = sum(
            (
                (self.mid_price - pos.entry_price) * pos.quantity
                if pos.quantity > 0
                else (pos.entry_price - self.mid_price) * (-pos.quantity)
            )
            for pos in self.positions
        )
        portfolio_value = self.cash + self.inventory * self.mid_price + unrealized
        return {
            "mid_price": self.mid_price,
            "inventory": self.inventory,
            "pending_orders": float(len(self._pending_limits)),
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "portfolio_value": portfolio_value,
            "step": float(self.step_count),
        }

    def _log_metrics(self) -> None:
        total_fills = len(self._fill_events)
        avg_slippage = self._slippage_paid / max(total_fills, 1)
        self.logger.info(
            "Execution summary | fills=%d | realized_pnl=%.4f | inventory=%.2f | avg_slippage=%.4f | spread_paid=%.4f",
            total_fills,
            self.realized_pnl,
            self.inventory,
            avg_slippage,
            self._spread_paid,
        )
        if total_fills:
            last_fill = self._fill_events[-1]
            self.logger.debug("Last fill: %s", last_fill)


__all__ = [
    "ExecutionConfig",
    "OrderAction",
    "SimulatedRetailExecutionEnv",
    "SlippageModel",
]
