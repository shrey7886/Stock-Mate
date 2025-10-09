"""
TFT Data Preparation Module
Handles TimeSeriesDataSet creation for Temporal Fusion Transformer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import yfinance as yf
from datetime import datetime, timedelta

class TFTDataPreprocessor:
    """Data preprocessing for Temporal Fusion Transformer model"""
    
    def __init__(self, 
                 max_prediction_length: int = 30,
                 max_encoder_length: int = 90,
                 batch_size: int = 64):
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.batch_size = batch_size
        
    def fetch_stock_data(self, 
                        symbols: List[str], 
                        start_date: str, 
                        end_date: str) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with stock data
        """
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    data['symbol'] = symbol
                    data['date'] = data.index
                    data = data.reset_index(drop=True)
                    all_data.append(data)
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                
        if not all_data:
            raise ValueError("No data fetched for any symbols")
            
        return pd.concat(all_data, ignore_index=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for TFT model
        
        Args:
            df: Raw stock data DataFrame
            
        Returns:
            Processed DataFrame with features
        """
        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date'])
        
        # Create time index
        df['time_idx'] = df.groupby('symbol').cumcount()
        
        # Calculate technical indicators
        df['sma_5'] = df.groupby('symbol')['Close'].rolling(5).mean().reset_index(0, drop=True)
        df['sma_20'] = df.groupby('symbol')['Close'].rolling(20).mean().reset_index(0, drop=True)
        df['rsi'] = self._calculate_rsi(df.groupby('symbol')['Close'])
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df.groupby('symbol')['Close'])
        
        # Calculate returns
        df['returns'] = df.groupby('symbol')['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df.groupby('symbol')['Close'].shift(1))
        
        # Volume features
        df['volume_sma'] = df.groupby('symbol')['Volume'].rolling(10).mean().reset_index(0, drop=True)
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price features
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def create_timeseries_dataset(self, 
                                df: pd.DataFrame,
                                target_column: str = 'Close',
                                time_varying_known_reals: List[str] = None,
                                time_varying_unknown_reals: List[str] = None) -> TimeSeriesDataSet:
        """
        Create TimeSeriesDataSet for TFT training
        
        Args:
            df: Processed DataFrame
            target_column: Target variable column
            time_varying_known_reals: Known real-valued time-varying features
            time_varying_unknown_reals: Unknown real-valued time-varying features
            
        Returns:
            TimeSeriesDataSet object
        """
        if time_varying_known_reals is None:
            time_varying_known_reals = ['Open', 'High', 'Low', 'Volume']
            
        if time_varying_unknown_reals is None:
            time_varying_unknown_reals = [
                'sma_5', 'sma_20', 'rsi', 'bb_upper', 'bb_lower',
                'returns', 'log_returns', 'volume_ratio', 'high_low_ratio', 'close_open_ratio'
            ]
        
        # Create the dataset
        training = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=target_column,
            group_ids=["symbol"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        return training
    
    def prepare_prediction_data(self, 
                              df: pd.DataFrame, 
                              last_n_days: int = 90) -> pd.DataFrame:
        """
        Prepare data for making predictions
        
        Args:
            df: Processed DataFrame
            last_n_days: Number of recent days to use for prediction
            
        Returns:
            DataFrame ready for prediction
        """
        # Get the most recent data for each symbol
        latest_data = df.groupby('symbol').tail(last_n_days)
        
        # Ensure we have enough data points
        latest_data = latest_data.groupby('symbol').filter(
            lambda x: len(x) >= self.max_encoder_length
        )
        
        return latest_data
