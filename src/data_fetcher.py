import os
import json
import random
import math
import re
import time
import yaml
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _resolve_env(value):
    if isinstance(value, str):
        matches = re.findall(r'\$\{(\w+)\}', value)
        for m in matches:
            value = value.replace(f'${{{m}}}', os.environ.get(m, ''))
    return value

class DataFetcher:
    def __init__(self, config_path=os.path.join(PROJECT_ROOT, 'config', 'config.yaml'), use_mock=True):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        for k, v in self.config.get('api', {}).items():
            self.config['api'][k] = _resolve_env(v)
        self.use_mock = use_mock
        if not self.use_mock:
            self._init_broker()
        self._stock_list_cache = None
        self._stock_list_cache_ts = 0

    def _get_secret(self, key, default=None):
        try:
            import streamlit as st
            return st.secrets[key]
        except Exception:
            return os.environ.get(key, default)

    def _init_broker(self):
        broker = self.config['api']['broker']
        if broker == 'fyers':
            from fyers_apiv3 import fyersModel
            self.fyers = fyersModel.FyersModel(
                client_id=self._get_secret('FYERS_CLIENT_ID', self.config['api'].get('client_id')),
                token=self._get_secret('FYERS_ACCESS_TOKEN', self.config['api'].get('access_token')),
                is_async=False,
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

    def _retry_call(self, fn, *args, retries=3, backoff=2, **kwargs):
        for attempt in range(retries):
            try:
                res = fn(*args, **kwargs)
                if isinstance(res, dict) and res.get('s') == 'error' and res.get('code') == 429:
                    time.sleep(backoff ** attempt)
                    continue
                return res
            except Exception:
                time.sleep(backoff ** attempt)
        return {}

    def get_nse_stock_list(self):
        if self.use_mock:
            return [f'STOCK{i:03d}' for i in range(1, 151)]
        if self.config['api']['broker'] == 'fyers':
            if self._stock_list_cache is not None and (time.time() - self._stock_list_cache_ts) < 3600:
                return self._stock_list_cache
            url = "https://public.fyers.in/sym_details/NSE_CM.csv"
            df = pd.read_csv(url, header=None)
            symbol_col = None
            for col in df.columns:
                if df[col].astype(str).str.match(r'^NSE:[A-Z0-9]+-EQ$').any():
                    symbol_col = col
                    break
            if symbol_col is None:
                raise RuntimeError("Could not find symbol column in Fyers instruments CSV")
            symbols = df[symbol_col].dropna().astype(str).unique().tolist()
            symbols = [s for s in symbols if s.startswith('NSE:') and s.endswith('-EQ')]
            self._stock_list_cache = symbols
            self._stock_list_cache_ts = time.time()
            return symbols
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
                'volume': random.randint(50000, 300000),
                'ltq': random.randint(1000, 50000)
            }
        if self.config['api']['broker'] == 'fyers':
            data = {"symbols": symbol}
            res = self._retry_call(self.fyers.quotes, data=data)
            if res.get('s') != 'ok' or not res.get('d'):
                return {'ltp': 0, 'bid_price': 0, 'bid_qty': 0, 'ask_price': 0, 'ask_qty': 0, 'volume': 0, 'ltq': 0}
            v = res['d'][0].get('v', {})
            return {
                'ltp': v.get('lp', 0),
                'bid_price': v.get('bid', 0),
                'bid_qty': v.get('volume', 0),
                'ask_price': v.get('ask', 0),
                'ask_qty': v.get('volume', 0),
                'volume': v.get('volume', 0),
                'ltq': v.get('ltq', 0)
            }
        data = {"exchange": "NSE", "symboltoken": symbol, "tradingsymbol": symbol}
        return self.smart.ltp(data)['data']

    def get_quotes_batch(self, symbols):
        if self.use_mock:
            return {sym: self.get_quote(sym) for sym in symbols}
        if self.config['api']['broker'] == 'fyers':
            result = {}
            for i in range(0, len(symbols), 50):
                chunk = symbols[i:i+50]
                data = {"symbols": ",".join(chunk)}
                res = self._retry_call(self.fyers.quotes, data=data)
                if res.get('s') == 'ok' and res.get('d'):
                    for item in res['d']:
                        v = item.get('v', {})
                        result[item.get('n', '')] = {
                            'ltp': v.get('lp', 0),
                            'bid_price': v.get('bid', 0),
                            'bid_qty': v.get('volume', 0),
                            'ask_price': v.get('ask', 0),
                            'ask_qty': v.get('volume', 0),
                            'volume': v.get('volume', 0),
                            'ltq': v.get('ltq', 0)
                        }
            return result
        return {sym: self.get_quote(sym) for sym in symbols}

    def get_market_depth(self, symbol):
        if self.use_mock:
            return self.get_quote(symbol)
        if self.config['api']['broker'] == 'fyers':
            data = {"symbol": symbol, "ohlcv_flag": "1"}
            res = self._retry_call(self.fyers.depth, data=data)
            if res.get('s') != 'ok' or not res.get('d'):
                return {'bid_price': 0, 'bid_qty': 0, 'ask_price': 0, 'ask_qty': 0}
            d = res['d'].get(symbol, next(iter(res['d'].values())) if isinstance(res['d'], dict) else res['d'][0])
            bids = d.get('bids', [])
            asks = d.get('ask', [])
            best_bid = bids[0] if bids else {'price': 0, 'volume': 0}
            best_ask = asks[0] if asks else {'price': 0, 'volume': 0}
            return {
                'bid_price': best_bid.get('price', 0),
                'bid_qty': best_bid.get('volume', 0),
                'ask_price': best_ask.get('price', 0),
                'ask_qty': best_ask.get('volume', 0)
            }
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
                "symbol": symbol,
                "resolution": "1",
                "date_format": "1",
                "range_from": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
                "range_to": datetime.now().strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            res = self._retry_call(self.fyers.history, data=data)
            df = pd.DataFrame(res.get('candles', []), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['ltp'] = df['close']
            return df
        hist = self.smart.get_candle_data(symbol, 'NSE', 'ONE_MINUTE', 60)['data']
        df = pd.DataFrame(hist, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['ltp'] = df['close']
        return df

    def fetch_historical_batch(self, symbols, max_workers=5):
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(self.get_historical_data, sym): sym for sym in symbols}
            for future in as_completed(future_map):
                sym = future_map[future]
                try:
                    results[sym] = future.result()
                except Exception:
                    results[sym] = None
        return results

    def get_depth_data(self, symbol):
        if self.use_mock:
            q = self.get_quote(symbol)
            return {
                'ltq': random.randint(1000, 50000),
                'bid_price': q['bid_price'],
                'bid_qty': q['bid_qty'],
                'ask_price': q['ask_price'],
                'ask_qty': q['ask_qty']
            }
        if self.config['api']['broker'] == 'fyers':
            data = {"symbol": symbol, "ohlcv_flag": "1"}
            res = self._retry_call(self.fyers.depth, data=data)
            if res.get('s') != 'ok' or not res.get('d'):
                return {'ltq': 0, 'bid_price': 0, 'bid_qty': 0, 'ask_price': 0, 'ask_qty': 0}
            d = res['d'].get(symbol, next(iter(res['d'].values())) if isinstance(res['d'], dict) else res['d'][0])
            bids = d.get('bids', [])
            asks = d.get('ask', [])
            best_bid = bids[0] if bids else {'price': 0, 'volume': 0}
            best_ask = asks[0] if asks else {'price': 0, 'volume': 0}
            return {
                'ltq': d.get('ltq', 0),
                'bid_price': best_bid.get('price', 0),
                'bid_qty': best_bid.get('volume', 0),
                'ask_price': best_ask.get('price', 0),
                'ask_qty': best_ask.get('volume', 0)
            }
        return {'ltq': 0, 'bid_price': 0, 'bid_qty': 0, 'ask_price': 0, 'ask_qty': 0}
