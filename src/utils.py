import sys 
import os
from dataclasses import dataclass
import dill
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,accuracy_score,f1_score
import numpy as np
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging



def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb' )as file_obj:
            dill.dump(obj,file_obj)
        logging.info(f'File successfully save : {dir_path}')
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models,params):
    print("y_test unique values:", np.unique(y_test))
    logging.info('Started evaluating the models')

    try:
        model_results = {}

        for name, model in models.items():

            #para=params[list(models.keys())[name]]
            print(f"\n➡️ Evaluating Model: {name} ({type(model)})")

            # # Train
            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(x_train,y_train)

            # #model.fit(x_train,y_train)
            # model.set_params(**gs.best_params_)
            # model.fit(x_train,y_train)
            para = params[name]
    
            gs = GridSearchCV(model, para, cv=3, scoring='f1', verbose=0)
            gs.fit(x_train, y_train)

            model = gs.best_estimator_
            y_pred = model.predict(x_test)
            # Predict
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:,1] if hasattr(model, "predict_proba") else None
            #y_prob = best_model.predict_proba(x_test)[:,1] if hasattr(best_model, "predict_proba") else None

            # Metrics
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            logging.info("Classification Report:")
            logging.info(f'Classification Report :\n{classification_report(y_test, y_pred)}')

            # AUC Score Handling
            try:
                if y_prob is not None:
                    if y_prob.ndim > 1 and y_prob.shape[1] > 1:  # multiclass
                        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                    else:
                        auc = roc_auc_score(y_test, y_prob)
                else:
                    auc = None
            except Exception as e:
                print(f"⚠️ AUC computation failed: {e}")
                auc = None


            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            model_results[name] = {
                "accuracy": acc,
                "f1_score": f1,
                "roc_auc": auc
            }

        return model_results

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):

    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)  
