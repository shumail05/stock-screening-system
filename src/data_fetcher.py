import os
import json
import random
import math
import yaml
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataFetcher:
    def __init__(self, config_path=os.path.join(PROJECT_ROOT, 'config', 'config.yaml'), use_mock=True):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.use_mock = use_mock
        if not self.use_mock:
            self._init_broker()

    def _get_secret(self, key, default=None):
        try:
            import streamlit as st
            return st.secrets[key]
        except Exception:
            return os.environ.get(key, default)

    def _init_broker(self):
        broker = self.config['api']['broker']
        if broker == 'fyers':
            import fyers_apiv2
            self.fyers = fyers_apiv2.FyersModel(
                client_id=self._get_secret('FYERS_CLIENT_ID', self.config['api'].get('client_id')),
                token=self._get_secret('FYERS_ACCESS_TOKEN', self.config['api'].get('access_token')),
                log_path=''
            )
        elif broker == 'angelone':
            from smartapi import SmartConnect
            api_key = self._get_secret('ANGELONE_API_KEY', self.config['api'].get('api_key'))
            client_id = self._get_secret('ANGELONE_CLIENT_ID', self.config['api'].get('client_id'))
            password = self._get_secret('ANGELONE_PASSWORD', self.config['api'].get('password'))
            pin = self._get_secret('ANGELONE_PIN', self.config['api'].get('pin'))
            self.smart = SmartConnect(api_key=api_key)
            session = self.smart.generateSession(client_id, password, pin)
            self.smart.access_token = session['data']['jwtToken']
            self.feed_token = self.smart.getfeedToken()['data']['feedToken']

    def get_nse_stock_list(self):
        if self.use_mock:
            return [f'STOCK{i:03d}' for i in range(1, 151)]
        if self.config['api']['broker'] == 'fyers':
            symbols = self.fyers.get_market_status()
            return [s['symbol'] for s in symbols if 'NSE' in s.get('exchange', '')]
        return [s['symboltoken'] for s in self.smart.getmaster()['data'] if s['exch_seg'] == 'NSE']

    def get_quote(self, symbol):
        if self.use_mock:
            base = 30 + (hash(symbol) % 470)
            return {
                'symbol': symbol,
                'ltp': round(base + random.uniform(-1, 1), 2),
                'bid_price': round(base - 0.05, 2),
                'bid_qty': random.randint(800000, 1500000),
                'ask_price': round(base + 0.05, 2),
                'ask_qty': random.randint(800000, 1500000),
                'volume': random.randint(50000, 300000)
            }
        if self.config['api']['broker'] == 'fyers':
            data = {"symbols": f"NSE:{symbol}-EQ"}
            return self.fyers.get_quotes(data=data)['d'][0]
        data = {"exchange": "NSE", "symboltoken": symbol, "tradingsymbol": symbol}
        return self.smart.ltp(data)['data']

    def get_market_depth(self, symbol):
        if self.use_mock:
            return self.get_quote(symbol)
        if self.config['api']['broker'] == 'fyers':
            data = {"symbol": f"NSE:{symbol}-EQ"}
            return self.fyers.get_depth(data=data)
        return self.smart.get_market_depth(symbol, 'NSE')['data']

    def get_historical_data(self, symbol, period='60D'):
        if self.use_mock:
            base = 30 + (hash(symbol) % 470)
            now = datetime.now()
            rows = 3000
            timestamps = [now - timedelta(minutes=i) for i in range(rows, 0, -1)]
            prices = [base + 5 * math.sin(i / 60) + random.uniform(-0.5, 0.5) for i in range(rows)]
            volumes = [random.randint(10000, 200000) for _ in range(rows)]
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices,
                'high': [p + random.uniform(0, 0.5) for p in prices],
                'low': [p - random.uniform(0, 0.5) for p in prices],
                'close': prices,
                'volume': volumes,
                'ltp': prices
            })
            return df
        if self.config['api']['broker'] == 'fyers':
            data = {
                "symbol": f"NSE:{symbol}-EQ",
                "resolution": "1",
                "date_format": "1",
                "range_from": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
                "range_to": datetime.now().strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            res = self.fyers.get_history(data=data)
            df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['ltp'] = df['close']
            return df
        hist = self.smart.get_candle_data(symbol, 'NSE', 'ONE_MINUTE', 60)['data']
        df = pd.DataFrame(hist, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['ltp'] = df['close']
        return df
