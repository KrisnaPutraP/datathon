import pathlib
from pathlib import Path

ROOT_DIR    = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "dataset"          
MODEL_DIR   = Path("/content/drive/MyDrive/datathon/models")  

TRAIN_FILE  = DATA_DIR / "copy_train.jsonl"
VALID_FILE  = DATA_DIR / "copy_valid.jsonl"

BASE_MODEL_NAME = "mistral-7b-instruct-v0.2"
BASE_MODEL_PATH = MODEL_DIR / BASE_MODEL_NAME

OUTPUT_DIR  = MODEL_DIR / "mistral-copywriter-lora-v6"

LR = 3e-5
BATCH_SIZE = 1
GRAD_ACCUM = 32
EPOCHS = 5
MAX_SEQ_LEN = 512
VALID_SPLIT = 0.1