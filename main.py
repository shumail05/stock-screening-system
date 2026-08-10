import os
import logging
from src.screener import StockScreener
from src.dashboard import Dashboard

def main():
    logging.basicConfig(level=logging.INFO)
    use_mock = True
    try:
        import streamlit as st
        has_angel = all(st.secrets.get(k) for k in ['ANGELONE_API_KEY', 'ANGELONE_CLIENT_ID', 'ANGELONE_PASSWORD', 'ANGELONE_PIN'])
        has_fyers = all(st.secrets.get(k) for k in ['FYERS_CLIENT_ID', 'FYERS_ACCESS_TOKEN'])
        use_mock = not (has_angel or has_fyers)
    except Exception:
        has_angel = all(os.environ.get(k) for k in ['ANGELONE_API_KEY', 'ANGELONE_CLIENT_ID', 'ANGELONE_PASSWORD', 'ANGELONE_PIN'])
        has_fyers = all(os.environ.get(k) for k in ['FYERS_CLIENT_ID', 'FYERS_ACCESS_TOKEN'])
        use_mock = not (has_angel or has_fyers)
    screener = StockScreener(use_mock=use_mock)
    dashboard = Dashboard(screener)
    dashboard.run()

if __name__ == '__main__':
    main()
