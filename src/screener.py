import os
import pandas as pd
from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
from src.smma_calculator import SMMACalculator
from src.filter_engine import FilterEngine
from src.analytics import AnalyticsEngine
from src.ml_predictor import MLPredictor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class StockScreener:
    def __init__(self, use_mock=True):
        config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
        self.data_fetcher = DataFetcher(config_path=config_path, use_mock=use_mock)
        self.smma_calc = SMMACalculator()
        self.filter_engine = FilterEngine(config_path=config_path)
        self.analytics = AnalyticsEngine()
        self.ml_predictor = MLPredictor()
        self.cache = {}

    def get_screened_stocks(self):
        symbols = self.data_fetcher.get_nse_stock_list()
        stocks_data = []
        for sym in symbols:
            quote = self.data_fetcher.get_quote(sym)
            stocks_data.append({
                'symbol': sym,
                'ltp': quote['ltp'],
                'bid_price': quote['bid_price'],
                'bid_qty': quote['bid_qty'],
                'ask_price': quote['ask_price'],
                'ask_qty': quote['ask_qty']
            })
        return self.filter_engine.apply_filters(stocks_data)

    def analyze_stock(self, symbol):
        if symbol in self.cache:
            hist = self.cache[symbol]
        else:
            hist = self.data_fetcher.get_historical_data(symbol)
            self.cache[symbol] = hist
        smma = self.smma_calc.detect_crossover(hist)
        latest_bullish = bool(smma['bullish_signal'].iloc[-1])
        latest_bearish = bool(smma['bearish_signal'].iloc[-1])
        qty5 = self.analytics.calculate_exchange_quantity(hist, 5)
        qty20 = self.analytics.calculate_exchange_quantity(hist, 20)
        qty60 = self.analytics.calculate_exchange_quantity(hist, 60)
        avg20 = self.analytics.calculate_avg_ltp(hist, 20)
        avg60 = self.analytics.calculate_avg_ltp(hist, 60)
        pred = self.ml_predictor.predict_signal_quality(hist)
        return {
            'symbol': symbol,
            'ltp': round(float(hist['ltp'].iloc[-1]), 2),
            'fast_smma': round(float(smma['fast_smma'].iloc[-1]), 2),
            'slow_smma': round(float(smma['slow_smma'].iloc[-1]), 2),
            'bullish_signal': latest_bullish,
            'bearish_signal': latest_bearish,
            'qty_5m': qty5,
            'qty_20m': qty20,
            'qty_60m': qty60,
            'avg_ltp_20m': avg20,
            'avg_ltp_60m': avg60,
            'ml_probability': pred['probability'],
            'ml_recommendation': pred['recommendation'],
            'ml_reason': pred['reason']
        }

    def get_active_signals(self, screened_symbols=None):
        if screened_symbols is None:
            screened_symbols = [s['symbol'] for s in self.get_screened_stocks()]
        results = []
        for sym in screened_symbols[:20]:
            results.append(self.analyze_stock(sym))
        return results
