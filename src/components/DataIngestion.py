import os
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from imblearn.over_sampling import SMOTE

from src.logger import logging
from src.exception import CustomException
from src.components.DataTransformation import DataTransformation
from src.components.DataTransformation import DataTransformationConfig
from src.components.ModelTrainer import ModelTrainers

@dataclass
class DataIngestionConfig():
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','Data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
       
    def DataPreProcessing(self):
         
        logging.info('Step 1: Enter into the Data Pre Processing')
        try:
            df=pd.read_csv('notebook/data/Fraud_Detection_Data_Usage.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            missing_info = df.isnull().sum().to_dict()
            logging.info(f"Features Missing values summary before handling the missing values: \n {missing_info}")
            
            if df.isnull().values.any():
                # Categorical fill
                for col in df.select_dtypes(include='object'):
                    df[col].fillna(df[col].mode()[0], inplace=True)

                # Numerical fill
                for col in df.select_dtypes(include=np.number):
                    df[col].fillna(df[col].median(), inplace=True)

                

                missing_info = df.isnull().sum().to_dict()
                logging.info(f"\nFeatures Missing values summary after handling the missing values: \n {missing_info}")
            
              

            else:
                logging.info("Step2: No Missing Values")
            

            logging.info('Parse datetime features')
            # df['START_TIME']=pd.to_datetime(df['START_TIME'],errors='coerce')
            # df['END_TIME']=pd.to_datetime(df['END_TIME'],errors='coerce')
            df['START_TIME'] = pd.to_datetime(df['START_TIME'], format="%y-%m-%d %I:%M:%S.%f %p %z", errors='coerce')
            df['END_TIME'] = pd.to_datetime(df['END_TIME'], format="%y-%m-%d %I:%M:%S.%f %p %z", errors='coerce')

            logging.info('Duration in seconds ')
            df['DURATION']=(df['END_TIME'] - df['START_TIME']).dt.total_seconds().fillna(0)

            logging.info('Extract hour from start time')
            df['START_HOUR'] = df['START_TIME'].dt.hour.fillna(-1)

            
            logging.info('Drop identifiers /Features [ID,CALLING_NUM,CALLED_NUMBER,START_TIME,END_TIME]')
            df.drop(columns=['ID', 'CALLING_NUM', 'CALLED_NUMBER', 'START_TIME', 'END_TIME'], inplace=True)



            logging.info(f"\n Data Frame shape:\n {df.shape}")
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()

            logging.info(f"\nDataFrame Info:\n{info_str}")
            correlation_matrix = df.corr(numeric_only=True)
            information=df.info()
            logging.info(f"\n Data Frame Correlation between Numric Features:\n {(correlation_matrix)}")

                # plt.figure(figsize=(10, 8))
                # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
                # plt.title('Correlation Matrix')
                # plt.show()

            for columns in df.columns:
               if columns == 'IS_FRAUD':
                    
                    logging.info(f'value conuts : {df[columns].value_counts()}')
            logging.info('Handle Class Imbalance')

            return df

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_ingestion(self,df):
        logging.info('Step 2: Enter into the Data Ingestion Menthon')
        try:
            #df=pd.read_csv('Fraud_Detection_Data_usage.csv')
            logging.info('Read the dataset as dataframe')

            # os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            # df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            
            
            logging.info('Spliting the data into Train and Test with 0.2%')


            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            #train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            train_set.to_csv(os.path.join(self.ingestion_config.train_data_path), index=False, header=True)
            test_set.to_csv(os.path.join(self.ingestion_config.test_data_path),index=False,header=True)
            
            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=='__main__':
    obj=DataIngestion()
    df=obj.DataPreProcessing()
    #obj.initiate_data_ingestion(df)
    train_data,test_data=obj.initiate_data_ingestion(df)
    data_transformation=DataTransformation()
    #data_transformation.initiate_data_trasnformation(train_data,test_data)
    train_arr,test_arr,_=data_transformation.initiate_data_trasnformation(train_data,test_data)

    modeltrainer =ModelTrainers()
    print(modeltrainer.Initiate_Model_Trainer(train_arr,test_arr))

    


    # obj.initiate_data_ingestion()
