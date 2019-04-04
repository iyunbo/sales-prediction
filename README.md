# Sales Prediction

### A machine learning project for Udacity's Machine Learning Engineer Nano Degree

#### Project description
A kaggle competition for sales forecasting

[Kaggle Project link](https://www.kaggle.com/c/rossmann-store-sales)

#### Project organization
![project structure](img/structure.png)
- **data directory**<br>
contains different dataset files for training and test from [project data page](https://www.kaggle.com/c/rossmann-store-sales/data), 
it also contains intermediate results: 
 
    - **features_x.txt** - list of features for training
    - **train.csv** - historical data including Sales
    - **test.csv** - historical data excluding Sales
    - **sample_submission.csv** - a sample submission file in the correct format
    - **store.csv** - supplemental information about the stores

- **notebooks directory**<br>
contains data visualization and analysis in form of jupyter notebook

- **scripts directory**<br>
contains python code for models training:

    - **2-run_basics.py** - run basic models without tuning for model comparison
    - **3-tune_random_forest.py** - tune random forest model and determine the optimized parameters (duration > 10h)
    - **3-tune_xgboost.py: tune** - xgboost model and determine the optimized parameters (duration > 10h)
    - **4-train_random_forest.py** - train random forest with the optimized (step 3) parameters
    - **4-train_xgboost.py** - train xgboost with the optimized (step 3) parameters
    - **5-evaluate.py** - make prediction with the persistent models (*.joblib, *.model) and evaluate the result
    - **6-final_model.py** - select the best model and train with complete dataset: train data + validation data
    - **models.py** - methods for models training and evaluation
    - **preparation.py** - data cleaning and data preparation code
    
- some big data files are in the google drive as following link:<br>
[Big files in Google Drive](https://drive.google.com/open?id=1J0LKDANYdk-bSciZjzH_GZN31PLY1mKv)

    - **feat_matrix.pkl**: persistence of feature matrix
    - ***.joblib**: persistence of models
    
#### Current scores
- Random Forest: 0.1475
- XGBoost: 0.151

#### Google Cloud
```
GOOGLE_APPLICATION_CREDENTIALS=C:\Users\yunbo\sales-prediction-190404234621.json
GOOGLE_CLOUD_PROJECT=sales-prediction-iyunbo
```