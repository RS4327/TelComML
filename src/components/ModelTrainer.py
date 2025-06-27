import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models



@dataclass
class ModelTrainingConfiguration:
    trained_model_path=os.path.join('artifacts','model.pkl')

class ModelTrainers:
    def __init__(self):
        self.model_trainer_configu=ModelTrainingConfiguration()
    def Initiate_Model_Trainer(self,train_arr,test_arr):
        try:
            logging.info('Split Train and Test input data')
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1].astype(int),
                train_arr[:,-1].astype(int),
                test_arr[:,:-1].astype(int),
                test_arr[:,-1].astype(int)
            
            )
            # print(f'Model x_train (not a DataFrame):\n{pd.DataFrame(x_train).head(2)}')
            # print(f'Model x_test (not a DataFrame):\n{pd.DataFrame(x_test).head(2)}')
            # print(f'Model y_train (not a DataFrame):\n{pd.DataFrame(y_train).head(2)}')
            # print(f'Model y_test (not a DataFrame):\n{pd.DataFrame(y_test).head(2)}')


            models={
                "LinearRegression":LogisticRegression(max_iter=1000),
                "DecisionClassifer":DecisionTreeClassifier(),
                "RandomForest":RandomForestClassifier(),
                "XGBoost":XGBClassifier()
                #"SVC":SVC(probability=True)
                #,"MLPClassifier":MLPClassifier(max_iter=500)
            }

            
            


            logging.info(f'Making the Imbalanced dataset to balanced dataset by using the SMOTE')
            # Apply SMOTE to the training data
            smote = SMOTE(random_state=42)
            x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
            #x_test_res, y_test_res = smote.fit_resample(x_test, y_test)

            logging.info('Starting the Hyper Parameter Tuning')
            params={
                "LinearRegression":{},
                'DecisionClassifer':{
                    'criterion':['gini','entropy','log_loss'],
                    'splitter':['best','random'],
                    #'max_depth':[5,15,20,None],
                    #'min_samples_split':[2,5,10],
                    'min_samples_leaf':[1,2,4],
                    #'max_features':[None,'sqrt','log2'],
                    'class_weight':[None,'balanced']

                },
                'RandomForest':{

                    'n_estimators': [100, 200, 300],                  # Number of trees
                    'criterion': ['gini', 'entropy', 'log_loss'],     # Split criteria
                    #'max_depth': [None, 10, 20, 30],                  # Max depth of tree
                    #'min_samples_split': [2, 5, 10],                  # Minimum samples to split a node
                    'min_samples_leaf': [1, 2, 4],                    # Minimum samples at a leaf node
                    #'max_features': ['sqrt', 'log2', None] ,          # Features to consider for best split
                    # 'bootstrap': [True, False],                       # Whether bootstrap samples are used
                    'class_weight': [None, 'balanced']                # Handle class imbalance

                },
                'XGBoost':{
                    'learning_rate': [0.01, 0.05, 0.1],               # Step size shrinkage
                    'n_estimators': [100, 200, 300],                  # Number of boosted trees
                    #'max_depth': [3, 5, 7, 10],                       # Max depth per tree
                    #'min_child_weight': [1, 3, 5],                    # Minimum sum of instance weight (Hessian) in a child
                    'gamma': [0, 0.1, 0.2],                           # Minimum loss reduction required to make a split
                    'subsample': [0.6, 0.8, 1.0],                     # Row sampling ratio per tree
                    'colsample_bytree': [0.6, 0.8, 1.0],              # Feature sampling ratio per tree
                    'scale_pos_weight': [1, 2, 5, 10],                # Used for class imbalance
                    #'reg_alpha': [0, 0.1, 1],                         # L1 regularization
                    #'reg_lambda': [1, 1.5, 2],                        # L2 regularization
                    #'objective': ['binary:logistic'],                 # Binary classification objective
                    #'use_label_encoder': [False],                     # Required for newer versions of XGBoost
                    'eval_metric': ['logloss']                        # Evaluation metric
                }
                               


            }



            model_report:dict=evaluate_models(
                x_train=x_train_res,
                y_train=y_train_res,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )
        
        # Find best model by F1 score or AUC
            #best_model_name = max(model_report, key=lambda name: model_report[name]["f1_score"])
            best_model_name = max(model_report, key=lambda name: 0.7 * model_report[name]["f1_score"] + 0.3 * model_report[name]["roc_auc"])

            best_model = models[best_model_name]
            best_score = (
                0.7 * model_report[best_model_name]["f1_score"] +
                0.3 * model_report[best_model_name]["roc_auc"]
            )

            logging.info(f"✅ Best Model: {best_model_name} with F1-score: {best_score:.3f}")

            # Save model
            best_model.fit(x_train, y_train)
            save_object(
                file_path=self.model_trainer_configu.trained_model_path,
                obj=best_model
            )

            return best_model

           

        
        except Exception as e:
            raise CustomException(e,sys)



