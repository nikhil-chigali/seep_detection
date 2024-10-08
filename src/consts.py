from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Data directory
DATA_DIR = ROOT_DIR / "data"

# YAML template
YAML_TEMPLATE = """# Train, val, test sets and data directory
path: "$data_dir$" # dataset root dir
train: images/train # train dir
val: images/val # val dir
test: images/test # test dir

# Classes 
names:
    0: Seep cls1
    1: Seep cls2
    2: Seep cls3
    3: Seep cls4
    4: Seep cls5
    5: Seep cls6
    6: Seep cls7
"""
