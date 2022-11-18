import pickle
import os
from src.data_handlers import data_handlers

with open(f"{os.getenv('MODEL_DIR')}../utils/provinces_and_alphabet_dict.pkl", "rb") as f:
    provinces_and_alphabet_dict: dict = pickle.load(f)

PROVINCES, ALPHABET, ALPHABET_NUMBERS = provinces_and_alphabet_dict.values()
