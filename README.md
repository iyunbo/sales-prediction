# sales-prediction

### A machine learning project for Udacity's Machine Learning Engineer NanoDegrees

#### Project description
A kaggle competition for sales forecasting

[Project link](https://www.kaggle.com/c/rossmann-store-sales)

#### Project structure
![project structure](img/structure.png)
- data directory

contains different dataset files for training and validation from project description (https://www.kaggle.com/c/rossmann-store-sales/data), 
it also contains intermediate results: 

    - feat_matrix.pkl: persistence of feature matrix
    - features_x.txt: list of features for training
    - *.joblib: persistence of models

- notebooks directory

data visualization and analysis in form of jupyter notebook

- scripts directory

python code for models training:

    - models.py: methods for training models
    - preperation.py: data cleaning and data preparation code
    - run_models.py: run basic models without tuning
    - train_random_forest.py: tune randome forest model and evaluate the result
    - train_xgboost.py: tune xgboost model and evaluate the result