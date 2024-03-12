# Car prices prediction

The project analyzes a dataset of cars with their characteristics and prices and uses the data train models for price prediction.

## Contents
### Notebooks
The notebooks in the `notebooks` folder are used for analysis, training models, and answering business questions.

- `01_explore_and_validate_data.ipynb` explores and visualizes data. It also validates and improves data quality.
- `02_make_features.ipynb` makes additional features for training.
- `03_xgb_trees_model.ipynb` trains an xgb regressor based on trees.
- `04_xgb_trees_model_tuned.ipynb` trains an xgb regressor based on trees after hyperparameter tuning.
- `05_xgb_trees_model_tuned_less_features.ipynb` trains an xgb regressor based on trees after hyperparameter tuning and removing `model_key`.
- `06_xgb_linear_model.ipynb` trains an xgb regressor based on linear models.
- `07_xgb_linear_model_tuned.ipynb` trains an xgb regressor based on linear models after hyperparameter tuning.
- `08_linear_model.ipynb` trains a simple linear regression.
- `09_answer_questions.ipynb` analyzes a model and answers business questions with it.

Notebooks numbered 01 and 02 are to be run first. Notebook numbered 09 can be run after each of the notebooks numbered 03 to 08, to answer business questions based on each model.

### Source code
The source code contains utility functions to support the work done in the notebooks.

### Requirements
This project uses Python 3.12.1 and poetry 1.7.1.

To run it, follow these steps:
- [Download Python](https://www.python.org/downloads/) (3.12) and install it.
- [Install Poetry](https://python-poetry.org/docs/#installing-with-pipx).
- Open a terminal and `cd` to the folder `cars-price` project folder.
- Poetry will create a virtual environment where to install the dependencies. If you want the virtual environment to be in the project's folder, run `poetry config virtualenvs.in-project true`.
- Install the dependencies by running `poetry install`.
