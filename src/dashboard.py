import streamlit as st
import pandas as pd
from datetime import datetime
import time

class Dashboard:
    def __init__(self, screener):
        self.screener = screener

    def run(self):
        st.set_page_config(page_title='AI/ML Stock Screener', layout='wide')
        st.title('AI/ML-Based Stock Market Screening System')
        with st.sidebar:
            st.header('Controls')
            auto_refresh = st.checkbox('Auto Refresh', value=True)
            refresh_interval = st.slider('Refresh Interval (sec)', 5, 60, 10)
        placeholder = st.empty()
        while True:
            with placeholder.container():
                self._display_main_table()
                self._display_signals_section()
            if auto_refresh:
                time.sleep(refresh_interval)
            else:
                break

    def _display_main_table(self):
        st.header('Screened Stocks')
        data = self.screener.get_screened_stocks()
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

    def _display_signals_section(self):
        st.header('SMMA Crossover Signals')
        signals = self.screener.get_active_signals()
        if not signals:
            st.info('No signals yet')
            return
        rows = []
        for s in signals:
            rows.append({
                'Symbol': s['symbol'],
                'LTP': s['ltp'],
                'Fast SMMA': s['fast_smma'],
                'Slow SMMA': s['slow_smma'],
                'Buy Signal': s['bullish_signal'],
                'Sell Signal': s['bearish_signal'],
                'Qty 5m': s['qty_5m'],
                'Qty 20m': s['qty_20m'],
                'Qty 60m': s['qty_60m'],
                'Avg LTP 20m': s['avg_ltp_20m'],
                'Avg LTP 60m': s['avg_ltp_60m'],
                'ML Probability': s['ml_probability'],
                'Recommendation': s['ml_recommendation'],
                'Reason': s['ml_reason'],
                'Buy Backtest': s.get('backtest_buy_profitable'),
                'Sell Backtest': s.get('backtest_sell_profitable'),
                'Backtest Reason': s.get('backtest_reason', '')
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
