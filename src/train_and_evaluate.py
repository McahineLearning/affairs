import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# def eval_metrics(actual, pred):

#     CM = confusion_matrix(y_test ,y_pred)
#     TN = CM[0][0]
#     FN = CM[1][0]
#     TP = CM[1][1]
#     FP = CM[0][1]
#     # true positive rate
#     TPR = TP/(TP+FN)
#     # Specificity or true negative rate
#     TNR = TN/(TN+FP) 
#     # Fall out or false positive rate
#     FPR = FP/(FP+TN)
#     # False negative rate
#     FNR = FN/(TP+FN)


#     print("TPR is", TPR)
#     print("TNR is", TNR)
#     print("FPR is", FPR)
#     print("FNR is", FNR)

#     accuracy = accuracy_score(y_test, y_pred)*100
#     f1_score_macro = f1_score(y_test, y_pred, average='macro')
#     f1_score_micro = f1_score(y_test, y_pred, average='micro')
#     f1_score_weighted = f1_score(y_test, y_pred, average='weighted')
#     roc_auc_score_macro = roc_auc_score(y_test, y_pred,average='macro')
#     roc_auc_score_macro = roc_auc_score(y_test, y_pred,average='micro')
#     roc_auc_score_weighted = roc_auc_score(y_test, y_pred,average='weighted')
#     precision = precision_score(y_test, y_pred)
#     recall_score = recall_score(y_test, y_pred)
#     error = 100-(accuracy_score(y_test, y_pred)*100)
#     clf_report = metrics.classification_report(y_test, y_pred)
#     return accuracy, f1_score_macro, f1_score_micro, f1_score_weighted, roc_auc_score_macro, roc_auc_score_micro, roc_auc_score_weighted, precision, recall_score, error, clf_report, TPR, TNR, FNR

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    

    C = config["estimators"]["LogisticRegression"]["params"]["C"]
    penalty = config["estimators"]["LogisticRegression"]["params"]["penalty"]
    solver =  config["estimators"]["LogisticRegression"]["params"]["solver"]

    target = [config["base"]["target_col"]]
    target = np.ravel(target)

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    y_train = np.ravel(train[target])
    y_test = np.ravel(test[target])

    X_train = train.drop(target, axis=1)
    X_test = test.drop(target, axis=1)

    lr = LogisticRegression(
        C=C, 
        penalty=penalty, 
        solver=solver)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    print(y_pred)

    CM = confusion_matrix(y_test ,y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)


    print("TPR is", TPR)
    print("TNR is", TNR)
    print("FPR is", FPR)
    print("FNR is", FNR)

    accuracy = accuracy_score(y_test, y_pred)*100
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    f1_score_micro = f1_score(y_test, y_pred, average='micro')
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted')
    roc_auc_score_macro = roc_auc_score(y_test, y_pred,average='macro')
    roc_auc_score_micro = roc_auc_score(y_test, y_pred,average='micro')
    roc_auc_score_weighted = roc_auc_score(y_test, y_pred,average='weighted')
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    error = 100-(accuracy_score(y_test, y_pred)*100)
    clf_report = classification_report(y_test, y_pred)


    print("LogisticRegression model ", " C :", C)
    print("  accuracy: %s" % accuracy)
    print("  f1_score_macro: %s" % f1_score_macro)
    print("  roc_auc_score_macro: %s" % roc_auc_score_macro)

####################################################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "accuracy": accuracy,
            "f1_score_macro": f1_score_macro,
            "roc_auc_score_macro": roc_auc_score_macro,
            "f1_score_micro": f1_score_micro,
            "f1_score_weighted": f1_score_weighted,
            "roc_auc_score_micro": roc_auc_score_micro,
            "roc_auc_score_weighted": roc_auc_score_weighted,
            "precision" : precision,
            "recall_score" : recall,
            "error" : error,
            "TPR" : TPR,
            "TNR": TNR,
            "FNR" : FNR
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "C": C,
            "penalty": penalty,
            "solver" : solver
        }
        json.dump(params, f, indent=4)
#####################################################


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

