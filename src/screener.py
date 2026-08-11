import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
from src.smma_calculator import SMMACalculator
from src.filter_engine import FilterEngine
from src.analytics import AnalyticsEngine
from src.ml_predictor import MLPredictor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'model.pkl')

class StockScreener:
    def __init__(self, use_mock=True):
        config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
        self.data_fetcher = DataFetcher(config_path=config_path, use_mock=use_mock)
        self.smma_calc = SMMACalculator()
        self.filter_engine = FilterEngine(config_path=config_path)
        self.analytics = AnalyticsEngine()
        self.ml_predictor = MLPredictor()
        self.cache = {}
        self._load_or_train_ml_model()

    def _load_or_train_ml_model(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                self.ml_predictor = pickle.load(f)
            return
        if self.data_fetcher.use_mock:
            return
        symbols = self.data_fetcher.get_nse_stock_list()[:5]
        hist_map = self.data_fetcher.fetch_historical_batch(symbols, max_workers=3)
        hist_data = [df for df in hist_map.values() if df is not None and len(df) > 140]
        if hist_data:
            combined = pd.concat(hist_data, ignore_index=True)
            try:
                self.ml_predictor.train(combined)
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump(self.ml_predictor, f)
            except Exception:
                pass

    def get_screened_stocks(self):
        symbols = self.data_fetcher.get_nse_stock_list()
        quotes = self.data_fetcher.get_quotes_batch(symbols)
        stocks_data = []
        for sym in symbols:
            q = quotes.get(sym, {'ltp': 0, 'bid_price': 0, 'bid_qty': 0, 'ask_price': 0, 'ask_qty': 0, 'volume': 0})
            if self.filter_engine.apply_filters([{
                'symbol': sym,
                'ltp': q['ltp'],
                'bid_qty': q['bid_qty'],
                'ask_qty': q['ask_qty']
            }]):
                stocks_data.append({
                    'symbol': sym,
                    'ltp': q['ltp'],
                    'bid_price': q['bid_price'],
                    'bid_qty': q['bid_qty'],
                    'ask_price': q['ask_price'],
                    'ask_qty': q['ask_qty']
                })
        return stocks_data

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
        crossover_result = self.ml_predictor.backtest_crossover(hist)
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
            'ml_reason': pred['reason'],
            'backtest_buy_profitable': crossover_result.get('buy_profitable'),
            'backtest_sell_profitable': crossover_result.get('sell_profitable'),
            'backtest_reason': crossover_result.get('reason', '')
        }

    def get_active_signals(self, screened_symbols=None):
        if screened_symbols is None:
            screened_symbols = [s['symbol'] for s in self.get_screened_stocks()]
        hist_map = self.data_fetcher.fetch_historical_batch(screened_symbols[:20])
        results = []
        for sym, hist in hist_map.items():
            if hist is None or len(hist) == 0:
                continue
            self.cache[sym] = hist
            results.append(self.analyze_stock(sym))
        return results
