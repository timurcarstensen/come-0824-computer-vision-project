# import this file at the top of any training file before importing any model or dataset
# s.t. the necessary environment variables are set

# standard library imports
import pickle
import os

# local imports (i.e. our own code)
# data_handlers need to be imported s.t. the dataset is guaranteed to be in place and the necessary
# environment variables are set
# noinspection PyUnresolvedReferences
from . import data_handlers

# reading in the lists provinces, alphabet and alphabet_numbers from the pickle file
with open(f"{os.getenv('LOG_DIR')}../utils/provinces_and_alphabet_dict.pkl", "rb") as f:
    provinces_and_alphabet_dict: dict = pickle.load(f)

PROVINCES, ALPHABET, ALPHABET_NUMBERS = provinces_and_alphabet_dict.values()

# check if the wandb_key_file file exists in utils, if not raise a FileNotFoundError
if not os.path.isfile(f"{os.getenv('LOG_DIR')}../utils/wandb_key_file"):
    raise FileNotFoundError(
        "wandb_api_key file not found; please create it in src/utils/ and place your wandb api key in it"
    )

# reading the wandb API key stored in utils/wandb_key_file into the environment variable WANDB_API_KEY
with open(f"{os.getenv('LOG_DIR')}../utils/wandb_key_file", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read()

# check if the WANDB_API_KEY environment variable is empty, if so, raise a UserError
if os.getenv("WANDB_API_KEY") == "":
    raise UserWarning(
        "WANDB_API_KEY environment variable is empty, please check utils/wandb_key_file"
    )