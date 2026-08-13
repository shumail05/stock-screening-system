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
            if st.button('Retrain ML Model'):
                with st.spinner('Retraining model on previous day data...'):
                    self.screener._train_on_previous_day_data()
                st.success('Model retrained successfully')
        placeholder = st.empty()
        while True:
            with placeholder.container():
                self._display_main_table()
                self._display_signals_section()
                self._display_ml_analysis()
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
                'LTQ': s.get('ltq', 0),
                'ML Probability': s['ml_probability'],
                'Recommendation': s['ml_recommendation'],
                'Reason': s['ml_reason'],
                'Buy Backtest': s.get('backtest_buy_profitable'),
                'Sell Backtest': s.get('backtest_sell_profitable'),
                'Backtest Reason': s.get('backtest_reason', ''),
                'Avoided': s.get('eval_avoided', 0),
                'Profitable': s.get('eval_profitable', 0),
                'Losses': s.get('eval_losses', 0),
                'Avoidance %': s.get('eval_avoidance_rate', 0.0),
                'Profit %': s.get('eval_profit_rate', 0.0),
                'Loss %': s.get('eval_loss_rate', 0.0)
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    def _display_ml_analysis(self):
        st.header('AI/ML Analysis Results')
        summary = self.screener.get_ml_analysis_summary()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Model Training Date', summary.get('training_date', 'N/A'))
        with col2:
            st.metric('Model Accuracy', f"{summary.get('accuracy', 0):.2f}%")
        with col3:
            st.metric('Total Signals', summary.get('metrics', {}).get('total_signals', 0))
        with col4:
            st.metric('Avoidance Rate', f"{summary.get('metrics', {}).get('avoidance_rate', 0):.2f}%")
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric('Profitable Signals', summary.get('metrics', {}).get('profitable_signals', 0))
        with col6:
            st.metric('Loss Signals', summary.get('metrics', {}).get('loss_signals', 0))
        with col7:
            st.metric('Profit Rate', f"{summary.get('metrics', {}).get('profit_rate', 0):.2f}%")
        history = summary.get('history', [])
        if history:
            st.subheader('Training History')
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df, use_container_width=True)
