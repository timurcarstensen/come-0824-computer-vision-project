import pickle
import os
from src.data_handlers import data_handlers

with open(f"{os.getenv('MODEL_DIR')}../utils/provinces_and_alphabet_dict.pkl", "rb") as f:
    provinces_and_alphabet_dict: dict = pickle.load(f)

PROVINCES, ALPHABET, ALPHABET_NUMBERS = provinces_and_alphabet_dict.values()

with open(f"{os.getenv('MODEL_DIR')}../utils/wandb_key_file", "r") as f:
    WANDB_API_KEY = f.read()

os.environ["WANDB_API_KEY"] = WANDB_API_KEY
