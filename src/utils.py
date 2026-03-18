import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj) -> None:
    """
    Serializes and saves any Python object to a .pkl file.
    Creates the directory if it doesn't exist.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

        logging.info(f"Object saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Loads and deserializes a .pkl file back into a Python object.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at: {file_path}")

        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        logging.info(f"Object loaded from: {file_path}")

        return obj

    except Exception as e:
        raise CustomException(e, sys)