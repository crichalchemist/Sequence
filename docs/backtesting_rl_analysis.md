# Backtesting.py fit vs existing RL and execution stack

## Current RL and execution capabilities
- **A3C harness:** Multi-worker actor-critic training already wraps the Dignity CNN–LSTM–attention encoder. It handles asynchronous rollouts, shared Adam optimizer state, and gradient synchronization with value/policy losses plus entropy regularization. The harness expects a gym-like environment factory, so any backtesting-driven env would plug in via the existing factory hook.
- **Dignity feature encoder inside RL:** The RL stack reuses the ModelConfig-driven Dignity encoder (LSTM + Conv1d + attention) as a shared feature extractor before policy/value heads. This keeps RL aligned with the forecasting backbone but currently assumes fixed feature counts and time-major tensors from the data agent.
- **Retail execution simulator:** A gym-like execution environment exists with spread, slippage, limit-fill probability, decision lag, FIFO PnL, and logging. Rewards are portfolio value deltas, making it suitable for policy evaluation without exchange dependencies.
- **Risk gating:** A risk manager throttles logits or regression outputs when drawdown, volatility, spread, or no-trade windows are hit. This can be used during backtests to mimic live guardrails by clamping actions to flat/low exposure.

## What backtesting.py would add
- **Vectorized historical replay:** backtesting.py provides fast bar-by-bar playback with built-in equity tracking. It would give us deterministic offline rollouts using our prepared candle/intrinsic-time datasets without standing up the simulated execution layer when exchange realism is unnecessary.
- **Indicator/strategy hooks:** Strategies in backtesting.py can call our feature generators directly, letting us validate that feature pipelines (ATR/RSI/Bollinger/imbalance/volatility clusters) actually lift Sharpe under simple rules before training RL.
- **Charting and diagnostics:** The library includes equity/position visualizations that could complement our logging-only A3C loop. This improves explainability for model audits and hyperparameter sweeps.

## Integration path
1. **Environment adapter:** Implement a thin adapter that exposes a backtesting.py `Strategy` or vectorized runner as a gym-like environment (reset/step), feeding observations shaped like our A3C expects (batch-first float tensors). Map fills/slippage either to the existing simulated retail rules or to backtesting.py’s built-ins depending on mode.
2. **Execution realism toggle:** Keep the current `SimulatedRetailExecutionEnv` for slippage/FIFO realism, but add a `BacktestingEnv` for deterministic historical replay. Use the existing `env_factory` hook in `A3CAgent` to choose between them at runtime.
3. **Risk manager integration:** Wrap backtesting actions through `RiskManager.apply_classification_logits` or `apply_regression_output` before submitting to the adapter. Log gate reasons to align offline backtests with live risk posture.
4. **Data source:** Reuse prepared datasets (including intrinsic-time bars) as the price feed for backtesting.py. Ensure spread/commission normalization is applied consistently so reward magnitudes match those assumed by the retail simulator.

## Security and safety notes
- **Deterministic seeds:** When using backtesting.py, keep RNG seeds and data windows fixed inside the adapter so results are reproducible and resistant to tampering. The current A3C harness already guards against malformed env returns; keep similar checks in the adapter.
- **No credential exposure:** backtesting.py runs offline; avoid mixing it with MetaApi/live connectors in the same process to prevent accidental leakage of auth tokens in logs. Maintain separate config paths for live vs. offline runs.
- **Action validation:** Mirror the normalization in `OrderAction.normalized` (size rounding, side/action checks) inside the adapter to avoid accepting malformed or adversarial actions from experimental policies.

## Starting-equity considerations
- **Simulated retail env baseline:** The `SimulatedRetailExecutionEnv` now seeds cash from `ExecutionConfig.initial_cash`, which defaults to **$50,000** to mirror the demonstration funding level. Positions still start flat unless you preload inventory. Portfolio equity therefore begins at the configured bankroll and evolves with trading performance.
- **How to target a specific starting equity:** Override `ExecutionConfig(initial_cash=...)` (and optionally add an `initial_inventory` hook if you need preloaded positions) so `reset` initializes cash to your chosen bankroll. Backtesting.py adapters should mirror the same initial cash/inventory so offline equity curves match the simulated retail setup.
