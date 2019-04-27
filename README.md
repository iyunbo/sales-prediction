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
    - **submission.csv** - prediction results
    - **external** - external data (weather and store state)

- **notebooks directory**<br>
contains data visualization and analysis in form of jupyter notebook

- **scripts directory**<br>
contains python code for models training:

    - **1-quick_scoring.py** - run basic models without tuning for model comparison
    - **2-feature_selection.py** - select features which have negative impact on model performance
    - **3-tune_random_forest.py** - tune random forest model and determine the optimized parameters (duration > 10h)
    - **3-tune_xgboost.py:** - tune xgboost model and determine the optimized parameters (duration > 10h)
    - **4-train_random_forest.py** - train random forest with the optimized (step 3) parameters
    - **4-train_xgboost.py** - train xgboost with the optimized (step 3) parameters
    - **4-train_on_google.py** - tune xgboost parameters on Google AI Platform
    - **5-improve.py** - additional improvements on features in order to get better score, including external data integration, store based metrics
    - **5-submit_models.py** - summit trained models and save results into database 
    - **6-final_model.py** - select the best model ensemble and summit the result to Kaggle
    - **trainer directory** 
        - **models.py** - methods for models training and evaluation
        - **preparation.py** - data cleaning and data preparation code
        - **task.py** - ensemble training task for execution on Google AI Platform
        - **util.py** - utility function for Google Storage management
    
- some big data files are in the google drive as following link:<br>
[Big files in Google Drive](https://drive.google.com/open?id=1J0LKDANYdk-bSciZjzH_GZN31PLY1mKv)

    - **feat_matrix.pkl**: persistence of feature matrix
    - ***.joblib**: persistence of models
    
#### final score
- XGBoost: 0.11106

#### project report
Rossmann_project_report.pdf

#### Google Cloud Configuration
```
GOOGLE_APPLICATION_CREDENTIALS=C:\Users\yunbo\sales-prediction-190404234621.json
GOOGLE_CLOUD_PROJECT=sales-prediction-iyunbo
```