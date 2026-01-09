"""
Integration tests for the complete data collection pipeline.

These tests verify that different data sources work together correctly
without mocking the downloaders themselves (though APIs are still mocked).
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for end-to-end data collection workflow."""

    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for alignment testing."""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30, freq='D'),
            'open': [1.10 + i*0.001 for i in range(30)],
            'high': [1.11 + i*0.001 for i in range(30)],
            'low': [1.09 + i*0.001 for i in range(30)],
            'close': [1.105 + i*0.001 for i in range(30)],
            'volume': [1000000] * 30
        })

    @pytest.fixture
    def sample_trade_data(self):
        """Sample trade data from Comtrade."""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=12, freq='MS'),
            'trade_balance': [100000 + i*10000 for i in range(12)],
            'imports': [1000000 + i*50000 for i in range(12)],
            'exports': [900000 + i*40000 for i in range(12)],
            'currency_pair': ['EURUSD'] * 12
        })

    @pytest.fixture
    def sample_economic_data(self):
        """Sample economic data from FRED."""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=365, freq='D'),
            'value': [4.33 + i*0.01 for i in range(365)],
            'series_id': ['FEDFUNDS'] * 365,
            'series_name': ['Federal Funds Rate'] * 365,
            'currency': ['USD'] * 365,
            'indicator_type': ['interest_rate'] * 365,
            'currency_pair': ['EURUSD'] * 365
        })

    @pytest.fixture
    def sample_shock_data(self):
        """Sample monetary policy shock data from ECB."""
        return pd.DataFrame({
            'date': pd.date_range('1999-01-04', periods=12, freq='Y'),
            'pc1': [0.12, -0.05, 0.03, -0.02, 0.08, 0.01, -0.03, 0.04, 
                    -0.01, 0.06, 0.02, -0.04],
            'STOXX50': [0.23, -0.10, 0.15, -0.08, 0.20, 0.05, -0.12, 0.18, 
                        -0.06, 0.22, 0.08, -0.14],
            'MP_median': [0.095, -0.025, 0.019, -0.009, 0.058, 0.005, 
                          -0.021, 0.033, -0.004, 0.045, 0.012, -0.031],
            'CBI_median': [0.028, -0.015, 0.012, -0.008, 0.015, 0.002, 
                           -0.009, 0.012, -0.003, 0.018, 0.005, -0.011],
            'currency_pair': ['EURUSD'] * 12
        })

    @patch('data.extended_data_collection.get_trade_indicators_for_forex')
    @patch('data.extended_data_collection.download_multiple_series')
    @patch('data.extended_data_collection.load_ecb_shocks_daily')
    def test_full_pipeline_eurusd(self, mock_shocks, mock_econ, mock_trade,
                                  sample_trade_data, sample_economic_data, 
                                  sample_shock_data):
        """Test complete pipeline for EURUSD fundamental data collection."""
        mock_trade.return_value = sample_trade_data
        mock_econ.return_value = sample_economic_data
        mock_shocks.return_value = sample_shock_data
        
        from data.extended_data_collection import collect_all_forex_fundamentals
        
        result = collect_all_forex_fundamentals("EURUSD", "2023-01-01", "2023-12-31")
        
        # Verify structure
        assert isinstance(result, dict)
        assert 'trade_data' in result
        assert 'economic_data' in result
        assert 'monetary_shocks' in result
        
        # Verify all data sources were called
        assert mock_trade.called
        assert mock_econ.called
        assert mock_shocks.called

    @patch('data.extended_data_collection.get_trade_indicators_for_forex')
    @patch('data.extended_data_collection.download_multiple_series')
    def test_data_merge_alignment_on_dates(self, mock_econ, mock_trade, 
                                          sample_price_data, sample_trade_data,
                                          sample_economic_data):
        """Test that different data sources align correctly on dates."""
        mock_trade.return_value = sample_trade_data
        mock_econ.return_value = sample_economic_data
        
        from data.extended_data_collection import merge_with_price_data
        
        # Merge trade data with price data
        result = merge_with_price_data(sample_price_data, sample_trade_data)
        
        # Verify alignment
        assert not result.empty
        assert 'date' in result.columns
        # Should have dates that exist in both datasets
        common_dates = set(sample_price_data['date']) & set(sample_trade_data['date'])
        if common_dates:
            assert any(result['date'].isin(common_dates))

    @pytest.mark.parametrize("currency_pair,start_date,end_date", [
        ("EURUSD", "2023-01-01", "2023-12-31"),
        ("GBPUSD", "2022-06-01", "2023-06-30"),
        ("USDJPY", "2023-01-01", "2023-03-31"),
    ])
    @patch('data.extended_data_collection.get_trade_indicators_for_forex')
    @patch('data.extended_data_collection.download_multiple_series')
    @patch('data.extended_data_collection.load_ecb_shocks_daily')
    def test_pipeline_multiple_pairs_and_dates(self, mock_shocks, mock_econ, mock_trade,
                                               currency_pair, start_date, end_date):
        """Test pipeline works for various currency pairs and date ranges."""
        mock_trade.return_value = pd.DataFrame()
        mock_econ.return_value = pd.DataFrame()
        mock_shocks.return_value = pd.DataFrame()
        
        from data.extended_data_collection import collect_all_forex_fundamentals
        
        result = collect_all_forex_fundamentals(currency_pair, start_date, end_date)
        
        assert isinstance(result, dict)
        assert all(key in result for key in ['trade_data', 'economic_data', 'monetary_shocks'])

    @patch('data.extended_data_collection.get_trade_indicators_for_forex')
    @patch('data.extended_data_collection.download_multiple_series')
    @patch('data.extended_data_collection.load_ecb_shocks_daily')
    def test_partial_data_availability(self, mock_shocks, mock_econ, mock_trade):
        """Test pipeline handles when some data sources are unavailable."""
        # Trade data available, economic data empty, shocks error
        mock_trade.return_value = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'trade_balance': [100000]
        })
        mock_econ.return_value = pd.DataFrame()  # Empty
        mock_shocks.side_effect = FileNotFoundError("Shocks file not found")
        
        from data.extended_data_collection import collect_all_forex_fundamentals
        
        result = collect_all_forex_fundamentals("EURUSD", "2023-01-01", "2023-12-31")
        
        # Should still return partial results
        assert isinstance(result, dict)
        assert not result['trade_data'].empty
        assert result['economic_data'].empty
        # Verify shocks error was handled - should have empty DataFrame or None
        assert 'monetary_shocks' in result
        assert result['monetary_shocks'] is None or result['monetary_shocks'].empty

    @patch('data.extended_data_collection.get_trade_indicators_for_forex')
    @patch('data.extended_data_collection.download_multiple_series')
    def test_cognee_format_integration(self, mock_econ, mock_trade, sample_trade_data):
        """Test that data is properly formatted for Cognee ingestion."""
        mock_trade.return_value = sample_trade_data
        mock_econ.return_value = pd.DataFrame()
        
        from data.extended_data_collection import collect_trade_data
        
        result = collect_trade_data("EURUSD", 2023, 2023, "test_key")
        
        # Check if formatted for Cognee - assert column exists
        assert 'full_text' in result.columns, "Expected 'full_text' column in Cognee-formatted data"
        assert all(isinstance(text, str) for text in result['full_text'])
        assert all(len(text) > 0 for text in result['full_text'])

    def test_data_types_preserved_through_pipeline(self, sample_price_data, sample_trade_data):
        """Test that data types are preserved through collection and merge."""
        from data.extended_data_collection import merge_with_price_data
        
        result = merge_with_price_data(sample_price_data, sample_trade_data)
        
        # Assert result is non-empty and has expected columns
        assert not result.empty, "merge_with_price_data should return non-empty DataFrame"
        assert 'date' in result.columns
        assert 'close' in result.columns or 'trade_balance' in result.columns
        
        # Numeric columns should be numeric
        if 'close' in result.columns:
            assert pd.api.types.is_numeric_dtype(result['close'])
        if 'trade_balance' in result.columns:
            assert pd.api.types.is_numeric_dtype(result['trade_balance'])
        
        # Date column should be datetime
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    @patch('data.extended_data_collection.get_trade_indicators_for_forex')
    def test_error_logging_on_api_failure(self, mock_trade, caplog):
        """Test that API failures are properly logged."""
        import logging
        caplog.set_level(logging.ERROR)
        mock_trade.side_effect = ConnectionError("API unreachable")
        
        from data.extended_data_collection import collect_trade_data
        
        result = collect_trade_data("EURUSD", 2023, 2023)
        
        assert result.empty
        # Verify error was logged
        assert any("API unreachable" in record.message or "ConnectionError" in record.message 
                   for record in caplog.records if record.levelno >= logging.ERROR)

    def test_column_consistency_across_merge(self, sample_price_data, sample_trade_data):
        """Test that merged data has consistent columns."""
        from data.extended_data_collection import merge_with_price_data
        
        result = merge_with_price_data(sample_price_data, sample_trade_data)
        
        if not result.empty:
            # Should have columns from both datasets
            assert 'date' in result.columns
            # Should not have unexpected columns
            assert all(isinstance(col, str) for col in result.columns)


@pytest.mark.integration
@pytest.mark.slow
class TestDataPipelinePerformance:
    """Performance tests for data collection pipeline."""

    @pytest.mark.parametrize("num_rows", [100, 1000, 10000])
    def test_merge_performance(self, num_rows):
        """Test merge performance with various dataset sizes."""
        from data.extended_data_collection import merge_with_price_data
        import time
        
        price_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=num_rows, freq='D'),
            'close': [1.10 + i*0.01 for i in range(num_rows)]
        })
        
        fundamental_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=num_rows, freq='D'),
            'value': [100 + i for i in range(num_rows)]
        })
        
        # Measure execution time
        start_time = time.time()
        result = merge_with_price_data(price_data, fundamental_data)
        elapsed_time = time.time() - start_time
        
        # Performance assertions
        assert not result.empty
        # Should complete in reasonable time (scale with data size)
        max_time = 0.01 * (num_rows / 100)  # 0.01s per 100 rows baseline
        assert elapsed_time < max_time, f"Merge took {elapsed_time:.3f}s, expected < {max_time:.3f}s"


    @patch('data.extended_data_collection.get_trade_indicators_for_forex')
    @patch('data.extended_data_collection.download_multiple_series')
    @patch('data.extended_data_collection.load_ecb_shocks_daily')
    def test_full_pipeline_performance(self, mock_shocks, mock_econ, mock_trade):
        """Test full pipeline performance with realistic data sizes."""
        # Create large datasets
        mock_trade.return_value = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'trade_balance': [100000 + i*100 for i in range(1000)]
        })
        
        mock_econ.return_value = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5000, freq='6H'),
            'value': [4.33 + i*0.001 for i in range(5000)]
        })
        
        mock_shocks.return_value = pd.DataFrame({
            'date': pd.date_range('1999-01-04', periods=300, freq='M'),
            'MP_median': [0.05 * (i % 10) / 10 for i in range(300)]
        })
        
        from data.extended_data_collection import collect_all_forex_fundamentals
        
        result = collect_all_forex_fundamentals("EURUSD", "2020-01-01", "2025-12-31")
        
        assert isinstance(result, dict)


@pytest.mark.integration
class TestDataValidation:
    """Tests for data validation in the pipeline."""

    def test_trade_data_positive_amounts(self):
        """Test that trade amounts are reasonable."""
        from data.downloaders.comtrade_downloader import download_trade_balance
        
        with patch('data.downloaders.comtrade_downloader.previewGet', 
                   return_value={'data': [
                       {'period': '202301', 'importValue': 1000000, 'exportValue': 800000}
                   ]}):
            df = download_trade_balance("842", 2023, 2023)
            
            if not df.empty:
                assert (df['imports'] >= 0).all()
                assert (df['exports'] >= 0).all()

    def test_economic_data_cognee_formatting(self):
        """Test that economic indicators are properly formatted for Cognee."""
        from data.downloaders.fred_downloader import format_for_cognee
        
        df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'value': [4.33],
            'series_name': ['Federal Funds Rate'],
            'series_id': ['FEDFUNDS'],
            'currency': ['USD'],
            'indicator_type': ['interest_rate']
        })
        
        result = format_for_cognee(df)
        
        # Formatted data should have text representation
        assert 'full_text' in result.columns
        assert len(result['full_text'].iloc[0]) > 0
        
        # Value should be in reasonable range for interest rate (0-20%)
        assert 0 <= df['value'].iloc[0] <= 20

    def test_shock_classification_consistency(self):
        """Test that shock classifications are consistent."""
        from data.downloaders.ecb_shocks_downloader import classify_shock_type
        
        test_cases = [
            ({'MP_median': 0.05, 'CBI_median': 0.01}, "MP"),  # MP-dominant
            ({'MP_median': 0.01, 'CBI_median': 0.05}, "CBI"),  # CBI-dominant
            ({'MP_median': 0.0, 'CBI_median': 0.0}, "neutral"),  # Neutral
        ]
        
        for row_dict, expected_keyword in test_cases:
            row = pd.Series(row_dict)
            result = classify_shock_type(row)
            assert expected_keyword in result.lower()

    def test_date_alignment_in_merge(self):
        """Test that dates are properly aligned in merge operations."""
        from data.extended_data_collection import merge_with_price_data
        
        price_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'close': [1.10 + i*0.01 for i in range(10)]
        })
        
        fundamental_df = pd.DataFrame({
            'date': pd.date_range('2023-01-05', periods=5, freq='D'),
            'value': [100 + i for i in range(5)]
        })
        
        result = merge_with_price_data(price_df, fundamental_df)
        
        if not result.empty:
            # Test merge strategy: should use inner join (only common dates)
            # Common dates: 2023-01-05 through 2023-01-09 (5 days)
            expected_dates = pd.date_range('2023-01-05', periods=5, freq='D')
            result_dates = sorted(result['date'])
            
            # Verify merge kept only common dates (inner join behavior)
            assert len(result_dates) == 5, "Merge should use inner join strategy"
            for expected, actual in zip(expected_dates, result_dates):
                assert expected == actual
            
            # All result dates should have data from both sources
            result_dates_set = set(result['date'])
            price_dates_set = set(price_df['date'])
            fundamental_dates_set = set(fundamental_df['date'])
            
            # Verify inner join: all results in both sources
            assert result_dates_set.issubset(price_dates_set & fundamental_dates_set)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
