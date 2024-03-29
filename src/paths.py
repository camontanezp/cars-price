from pathlib import Path
import os

PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR /'raw'
CLEANED_DATA_DIR = DATA_DIR / 'cleaned'
TRANSFORMED_DATA_DIR = DATA_DIR / 'transformed'
MODELS_DIR = PARENT_DIR / 'models'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(CLEANED_DATA_DIR).exists():
    os.mkdir(CLEANED_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR) 

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)