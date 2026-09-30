from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
def load_config(): return yaml.safe_load((ROOT/'config/config.yaml').read_text())
