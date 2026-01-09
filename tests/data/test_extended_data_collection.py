"""
Integration tests for extended data collection.

Tests the unified interface for collecting all fundamental data sources.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.extended_data_collection import (
    collect_all_forex_fundamentals,
    merge_with_price_data
)


class TestCollectAllForexFundamentals:
    """Test unified fundamental data collection."""

    @patch("data.extended_data_collection.download_trade_balance")
    @patch("data.extended_data_collection.get_forex_economic_indicators")
    @patch("data.extended_data_collection.load_ecb_shocks")
    def test_collect_all_eurusd_success(
        self,
        mock_ecb,
        mock_fred,
        mock_comtrade
    ):
        """Test successful collection of all data for EUR/USD."""
        # Setup mocks
        mock_comtrade.return_value = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="MS"),
            "trade_balance": [1000, 2000, 3000],
            "exports": [5000, 6000, 7000],
            "imports": [4000, 4000, 4000]
        })

        mock_fred.return_value = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="ME"),
            "value": [4.5, 4.6, 4.7],
            "series_id": ["FEDFUNDS"] * 3,
            "series_name": ["Fed Funds Rate"] * 3,
            "currency": ["USD"] * 3,
            "indicator_type": ["interest_rate"] * 3
        })

        mock_ecb.return_value = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="D"),
            "MP_median": [0.001, -0.002, 0.003],
            "CBI_median": [0.002, 0.001, -0.001]
        })

        # Execute
        data = collect_all_forex_fundamentals(
            currency_pair="EURUSD",
            start_date="2023-01-01",
            end_date="2023-03-31",
            fred_api_key="test_key"
        )

        # Verify structure
        assert isinstance(data, dict)
        assert "trade" in data
        assert "economic" in data
        assert "shocks" in data

        # Verify data
        assert not data["trade"].empty
        assert not data["economic"].empty
        assert not data["shocks"].empty
        
        # Assert mocks were called with expected parameters using call_args.kwargs
        mock_comtrade.assert_called_once()
        assert mock_comtrade.call_args.kwargs["currency_pair"] == "EURUSD"
        
        mock_fred.assert_called_once()
        assert mock_fred.call_args.kwargs["currency_pair"] == "EURUSD"
        assert mock_fred.call_args.kwargs["fred_api_key"] == "test_key"
        # Verify exact values for start/end dates
        assert mock_fred.call_args.kwargs["start_date"] == "2023-01-01"
        assert mock_fred.call_args.kwargs["end_date"] == "2023-12-31"
        
        mock_ecb.assert_called_once()

    @patch("data.extended_data_collection.download_trade_balance")
    @patch("data.extended_data_collection.get_forex_economic_indicators")
    def test_collect_selective_sources(
        self,
        mock_fred,
        mock_comtrade
    ):
        """Test collecting only specified sources."""
        mock_comtrade.return_value = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=2, freq="ME"),
            "trade_balance": [1000, 2000]
        })

        # Execute - only trade data
        data = collect_all_forex_fundamentals(
            currency_pair="EURUSD",
            start_date="2023-01-01",
            end_date="2023-12-31",
            include_sources=["trade"]
        )

        # Verify only trade data collected
        assert "trade" in data
        assert not data["trade"].empty
        assert "economic" not in data or data["economic"].empty
        assert "shocks" not in data or data["shocks"].empty

    def test_collect_invalid_currency_pair(self):
        """Test handling of invalid currency pair."""
        with pytest.raises((ValueError, KeyError)):
            collect_all_forex_fundamentals(
                currency_pair="INVALID",
                start_date="2023-01-01",
                end_date="2023-12-31"
            )

    @patch("data.extended_data_collection.download_trade_balance")
    @patch("data.extended_data_collection.get_forex_economic_indicators")
    @patch("data.extended_data_collection.load_ecb_shocks")
    def test_collect_handles_partial_failures(
        self,
        mock_ecb,
        mock_fred,
        mock_comtrade
    ):
        """Test that partial failures don't break entire collection."""
        # Comtrade succeeds
        mock_comtrade.return_value = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=2, freq="MS"),
            "trade_balance": [1000, 2000]
        })

        # FRED fails
        mock_fred.side_effect = Exception("API error")

        # ECB succeeds
        mock_ecb.return_value = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=2, freq="D"),
            "MP_median": [0.001, -0.002]
        })

        # Execute - should still return data from successful sources
        data = collect_all_forex_fundamentals(
            currency_pair="EURUSD",
            start_date="2023-01-01",
            end_date="2023-12-31",
            fred_api_key="test_key"
        )

        # Should have trade and shocks, but not economic
        assert "trade" in data and not data["trade"].empty
        assert "shocks" in data and not data["shocks"].empty
        assert "economic" not in data or data["economic"].empty


class TestMergeWithPriceData:
    """Test merging fundamental data with price data."""

    def test_merge_with_price_data_forward_fill(self, sample_price_data):
        """Test that fundamental data is forward-filled to match price frequency."""
        # Create low-frequency fundamental data
        fundamental_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=2, freq="D"),
            "interest_rate": [4.5, 4.6],
            "trade_balance": [1000, 1100]
        })

        # Execute
        merged = merge_with_price_data(
            price_data=sample_price_data,
            fundamental_data=fundamental_data,
            date_column="timestamp"
        )

        # Verify
        assert len(merged) == len(sample_price_data)
        assert "interest_rate" in merged.columns
        assert "trade_balance" in merged.columns

        # Check forward fill worked
        assert merged["interest_rate"].notna().any()
        assert merged["trade_balance"].notna().any()

    def test_merge_preserves_price_columns(self, sample_price_data):
        """Test that all original price columns are preserved."""
        fundamental_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=1, freq="D"),
            "gdp": [25000]
        })

        merged = merge_with_price_data(
            price_data=sample_price_data,
            fundamental_data=fundamental_data,
            date_column="timestamp"
        )

        # All original columns should be present
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in merged.columns
            assert len(merged[col]) == len(sample_price_data)

    def test_merge_empty_fundamental_data(self, sample_price_data):
        """Test merge with empty fundamental data."""
        empty_fundamental = pd.DataFrame()

        merged = merge_with_price_data(
            price_data=sample_price_data,
            fundamental_data=empty_fundamental,
            date_column="timestamp"
        )

        # Should return original price data
        assert len(merged) == len(sample_price_data)
        pd.testing.assert_frame_equal(merged, sample_price_data)


@pytest.mark.parametrize("currency_pair", [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD"
])
@patch("data.extended_data_collection.download_trade_balance")
@patch("data.extended_data_collection.get_forex_economic_indicators")
@patch("data.extended_data_collection.load_ecb_shocks")
def test_collect_multiple_pairs(
    mock_ecb,
    mock_fred,
    mock_comtrade,
    currency_pair
):
    """Test that collection works for all major currency pairs."""
    # Setup minimal mocks
    mock_comtrade.return_value = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")]})
    mock_fred.return_value = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")]})
    mock_ecb.return_value = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")]})

    # Execute
    data = collect_all_forex_fundamentals(
        currency_pair=currency_pair,
        start_date="2023-01-01",
        end_date="2023-01-31",
        fred_api_key="test_key"
    )

    # Should return dictionary with data
    assert isinstance(data, dict)
    assert len(data) > 0
