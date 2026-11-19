"""python scripts/train_sft.py --config configs/sft.yaml   (GPU env only)"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lipogram.train_sft import train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/sft.yaml")
train(yaml.safe_load(Path(ap.parse_args().config).read_text(encoding="utf-8")))
