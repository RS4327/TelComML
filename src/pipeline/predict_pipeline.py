import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(
            
            self,
            CALL_TYPE:str,
            CHARGE:int,
            CALL_RESUL:str,
            CALL_TO:str,
            FEQ:int,
            DURATION:int,
            START_HOUR:int):
        self.CALL_TYPE=CALL_TYPE
        self.CHARGE=CHARGE
        self.CALL_RESUL=CALL_RESUL
        self.CALL_TO=CALL_TO
        self.FEQ=FEQ
        self.DURATION=DURATION
        self.START_HOUR=START_HOUR
    
    def get_data_as_data_frame(self):

        try:
             custom_data_input_dict={

                'CALL_TYPE':[self.CALL_TYPE],
                'CHARGE':[self.CHARGE],
                'CALL_RESUL':[self.CALL_RESUL],
                'CALL_TO':[self.CALL_TO],
                'FEQ':[self.FEQ],
                'DURATION':[self.DURATION],
                'START_HOUR':[self.START_HOUR]
            }
             return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
        




