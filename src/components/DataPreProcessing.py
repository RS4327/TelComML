import pandas as pd
import numpy as np
import sys
import os
#import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import logging
from src.exception import CustomException

 

class DataPreProcessing:
    def __init__(self):
        logging.info(f'Step1: Loaded data for Pre-Processing. Data shape:')
    def DataPreProcess(self):
        logging.info("Enter the Data Ingestion methond or component")
        try:
            # if not os.path.exists('notebook/data/Fraud_Detection_Data_Usage.csv'):
            #      raise FileNotFoundError("CSV file not found.")
            if  os.path.exists('notebook/data/Fraud_Detection_Data_Usage.csv'):
                print('Hello')
                #df=pd.read_csv('notebook/data/Fraud_Detection_Data_Usage.csv')
                #logging.info(f'Step1: Loaded data for Pre-Processing. Data shape: {df.shape}')
                logging.info(f'Step1: Loaded data for Pre-Processing. Data shape:')
            
           # print(df.head(1))


            
        except Exception as e:
            raise CustomException(e,sys)
        
            

if __name__=="__main":
    obj = DataPreProcessing()
    obj.DataPreProcess()
         