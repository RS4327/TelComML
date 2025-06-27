import os
import sys
from dataclasses import dataclass

import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import io 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocesser_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns = ['CHARGE', 'FEQ', 'DURATION', 'START_HOUR']
            categorical_columns = ['CALL_TYPE', 'CALL_RESUL', 'CALL_TO']
        
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler(with_mean=False))
            ])
            logging.info('Numerical columns: encoding & scaling completed')

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info('Categorical columns: encoding & scaling completed')
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_trasnformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('✅ Read Train and Test Data Completed')
            

            logging.info(f"\n Data Frame shape:\n {test_df.shape}")
            buffer = io.StringIO()
            test_df.info(buf=buffer)
            info_str = buffer.getvalue()

            logging.info(f"\nDataFrame Info-train_df before :\n{info_str}")
                        
            target_column = "IS_FRAUD"

            # 🔧 Step 1: Clean and convert target column
            logging.info("🧹 Cleaning target column to ensure binary classification")
            label_mapping = {

                'n': 0,
                'y': 1
                }
            train_df[target_column] = train_df[target_column].map(label_mapping)
            test_df[target_column] = test_df[target_column].map(label_mapping)
            train_df['IS_FRAUD_NEW']=train_df['IS_FRAUD']
            test_df['IS_FRAUD_NEW']=test_df['IS_FRAUD']
            
            print("y_test  before values: \n", train_df.head(2))
            print("y_test before values:\n", train_df.head(2))
            
            train_df.drop(columns='IS_FRAUD', axis=1, inplace=True)
            test_df.drop(columns='IS_FRAUD', axis=1, inplace=True)

            

            train_df.rename(columns={'IS_FRAUD_NEW': 'IS_FRAUD'}, inplace=True)
            test_df.rename(columns={'IS_FRAUD_NEW': 'IS_FRAUD'}, inplace=True)
            #✅ Check for empty DataFrames
            logging.info(f"\n Data Frame shape:\n {test_df.shape}")
            buffer = io.StringIO()
            test_df.info(buf=buffer)
            info_str = buffer.getvalue()

            logging.info(f"\nDataFrame Info-train_df:\n{info_str}")
            
            
            print("y_test after values:\n", train_df.head(2))
            print("y_test after values:\n", train_df.head(2))

            if train_df.empty or test_df.empty:

                raise CustomException("❌ After filtering, one of the DataFrames is empty. Check 'IS_FRAUD' class values!", sys)

             # Optional: convert if still string
            train_df[target_column] = train_df[target_column].astype(int)
            test_df[target_column] = test_df[target_column].astype(int)
            
            preprocessing_obj = self.get_data_transformation_object()

            logging.info('# 🔍 Step 3: Split features and target')

            input_features_train_df1 = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df1 = train_df[target_column]

            
            input_features_test_df1 = test_df.drop(columns=[target_column], axis=1)
            target_features_test_df1 = test_df[target_column]

            logging.info('# ⚙️ Step 4: Apply preprocessing')
            logging.info('Applying preprocessing object on training and testing dataframes')

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df1)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df1)
            
            logging.info('# 📦 Step 5: Combine with target')
            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df1)]
            test_arr = np.c_[input_features_test_arr, np.array(target_features_test_df1)]



            tt=pd.DataFrame(train_arr)
            ttt=pd.DataFrame(test_arr)
            
            print("y_test unique values train_arr :\n", tt.head(2))
            print("y_test unique values test_arr :\n", ttt.head(2))
            
            print("y_test unique values:", np.unique(test_arr[:,1].astype(int)))


            logging.info('Saved preprocessing object successfully')

            save_object(
                file_path=self.data_transformation_config.preprocesser_obj_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_path
            )

        except Exception as e:
            raise CustomException(e, sys)
