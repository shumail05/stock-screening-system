import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class FilterEngine:
    def __init__(self, config_path=os.path.join(PROJECT_ROOT, 'config', 'config.yaml')):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def apply_filters(self, stocks_data):
        cfg = self.config['screening']
        return [
            s for s in stocks_data
            if cfg['price_min'] <= s['ltp'] <= cfg['price_max']
            and s['bid_qty'] > cfg['min_bid_qty']
            and s['ask_qty'] > cfg['min_ask_qty']
        ]
