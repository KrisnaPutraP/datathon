import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"           # ↳ place your CSV/JSONL here
MODEL_DIR = ROOT_DIR / "models"           # ↳ store downloaded base model & checkpoints

TRAIN_FILE = DATA_DIR / "copy_train.jsonl"  # will be auto‑generated
VALID_FILE = DATA_DIR / "copy_valid.jsonl"

# Download the base model manually (offline) into MODEL_DIR
BASE_MODEL_NAME = "mistral-7b-instruct-v0.2"   # eg folder name after download
BASE_MODEL_PATH = MODEL_DIR / BASE_MODEL_NAME  # local dir (no internet)

# PEFT fine‑tune output
OUTPUT_DIR = MODEL_DIR / "mistral-copywriter-lora"

# Hyper‑params (edit freely)
LR              = 3e-5
BATCH_SIZE      = 2         # per device
GRAD_ACCUM      = 16        # effective BS 32
EPOCHS          = 3
MAX_SEQ_LEN     = 768       # tokens – small, enough for short prompt+response
VALID_SPLIT     = 0.1       # 10 % dev/val