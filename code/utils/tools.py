from collections import defaultdict

import os
import numpy as np
import pandas as pd
import random
import torch.utils.data as Data
import torch

from sklearn import linear_model
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score,r2_score, accuracy_score
from sklearn.metrics import f1_score,log_loss
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from .logger import error, info


from utils.reward import relative_absolute_error

seed_value = 0
np.random.seed(seed_value)
#torch.manual_seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)


def cube(x):
    return x ** 3

'''
def justify_operation_type(o):
    if o == 'sqrt':
        o = np.sqrt
    elif o == 'square':
        o = np.square
    elif o == 'sin':
        o = np.sin
    elif o == 'cos':
        o = np.cos
    elif o == 'tanh':
        o = np.tanh
    elif o == 'reciprocal':
        o = np.reciprocal
    elif o == '+':
        o = np.add
    elif o == '-':
        o = np.subtract
    elif o == '/':
        o = np.divide
    elif o == '*':
        o = np.multiply
    elif o == 'stand_scaler':
        o = StandardScaler()
    elif o == 'minmax_scaler':
        o = MinMaxScaler(feature_range=(-1, 1))
    elif o == 'quan_trans':
        o = QuantileTransformer(random_state=0)
    elif o == 'exp':
        o = np.exp
    elif o == 'cube':
        o = cube
    elif o == 'sigmoid':
        o = expit
    elif o == 'log':
        o = np.log
    else:
        print('Please check your operation!')
    return o
'''


def generate_action_mask(feature_data: np.ndarray, operation_list: list) -> list:
    """
    Returns a binary mask (list of 0/1) indicating valid or invalid operations 
    for each operation in operation_list given the feature_data.
    """
    mask = []
    for op in operation_list:
        if op == 'sqrt':
            valid = np.all(feature_data >= 0)
        elif op == 'reciprocal':
            valid = np.all(feature_data != 0)
        elif op == 'log':
            valid = np.all(feature_data > 0)
        else:
            valid = True
        mask.append(1 if valid else 0)
    return mask



def evaluate_features(env, feature_list, task_type, n_splits=2):
    """
    Evaluate the current feature set using cross-validation 
    and return a performance metric (mse or f1).
    """
    X = env.feature_df[env.current_features]
    y = env.target   
    if task_type == 'reg':     
        kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0) 
        mse_list = []
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            mse_list.append(mean_squared_error(y_test, y_predict))
        return np.mean(mse_list), np.mean(rae_list)
        #mse = np.mean(mse_list)
        #rae = np.mean(rae_list)
        

    elif task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        f1_list = []
        cross_entropy_list = []

        skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)


            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            cross_entropy_list.append(log_loss(y_test, y_proba))
        f1 = np.mean(f1_list)
        cross_entropy = np.mean(cross_entropy_list)
        return f1, cross_entropy
        
def test_task_new(feature, target, task_type, n_splits=2):
    X = feature
    y = target   
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list = [], [], []
        skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average=
            'weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted')
                            )
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list = [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list)
    else:
        return -1



def calculate_meta_statistics(feature_data,log_offset=1e-6):

    meta_stats = []
    for col in range(feature_data.shape[1]):  
        col_data = feature_data[:, col]
        meta_statistics = [
            np.mean(col_data),        # Mean of the column
            np.std(col_data),         # Standard deviation
            np.min(col_data),         # Minimum value
            np.max(col_data),         # Maximum value
            np.percentile(col_data, 25),  # Q1
            np.percentile(col_data, 50),  # Median (Q2)
            np.percentile(col_data, 75)   # Q3
        ]
        meta_stats.append(meta_statistics)

    meta_stats = np.array(meta_stats)  # Shape: (n_statistics, n_meta_statistics)
    flat = meta_stats.flatten()
    val_min = flat.min()
    val_max = flat.max()

    EPS = 1e-8
        
    # Convert to numpy array for consistent handling
    meta_stats_norm = (meta_stats - val_min) / (val_max - val_min + EPS)
    
    #return meta_stats
    return meta_stats_norm




def overall_feature_selection(best_features, task_type):
    if task_type == 'reg':
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].astype(int)
        reg = linear_model.Lasso(alpha=0.1).fit(X, y)
        model = SelectFromModel(reg, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        mae, mse, rae = test_task_new(new_data, task_type)
        info('mae: {:.3f}, mse: {:.3f}, 1-rae: {:.3f}'.format(mae, mse, 1 -
                                                              rae))
    elif task_type == 'cls':
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].astype(int)
        clf = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        acc, pre, rec, f1 = test_task_new(new_data, task_type)
        info('acc: {:.3f}, pre: {:.3f}, rec: {:.3f}, f1: {:.3f}'.format(acc,
                                                                        pre, rec, f1))
    return new_data

def create_op_seq(operations, operation_emb):

    op_tensors = []
    for op_name in operations:
        # Each op_name is e.g. "sqrt", "sin", ...
        emb_vec = operation_emb[op_name]  # shape [emb_dim]
        op_tensors.append(emb_vec.unsqueeze(0))  
        # unsqueeze(0) => shape [1, emb_dim] so we can cat easily

    # Stack along dim=0 => shape [seq_len, emb_dim]
    op_seq = torch.cat(op_tensors, dim=0)
    return op_seq


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error)



def select_topk_features_and_evaluate(
    feature_pool: pd.DataFrame,
    y: np.ndarray,
    k: int = 10,
    task_type: str = 'reg',
    n_splits: int = 5,
    random_state: int = 0
):


    if task_type == 'cls':
        rf_full = RandomForestClassifier(random_state=random_state)
    elif task_type == 'reg':
        rf_full = RandomForestRegressor(random_state=random_state)
    else:
        raise ValueError("task_type must be 'cls' or 'reg'")
    
    rf_full.fit(feature_pool, y)
    importances = rf_full.feature_importances_  
    
    sorted_idx = np.argsort(importances)[::-1]   
    topk_idx = sorted_idx[:k]
    top_k_features = feature_pool.columns[topk_idx]
    
    X_topk = feature_pool[top_k_features]
    
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=random_state)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        pre_list, rec_list, f1_list, acc_list = [], [], [], []
        for train_idx, test_idx in skf.split(X_topk, y):
            X_train, y_train = X_topk.iloc[train_idx, :], y[train_idx]
            X_test, y_test = X_topk.iloc[test_idx, :], y[test_idx]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            pre_list.append(precision_score(y_test, y_pred, average='weighted'))
            rec_list.append(recall_score(y_test, y_pred, average='weighted'))
            f1_list.append(f1_score(y_test, y_pred, average='weighted'))
            acc_list.append(accuracy_score(y_test, y_pred))
        
        metrics = {
            'mean_precision': np.mean(pre_list),
            'mean_recall':    np.mean(rec_list),
            'mean_f1':        np.mean(f1_list),
            'mean_accuracy':  np.mean(acc_list) 
        }
    
    else:  
        reg = RandomForestRegressor(random_state=random_state)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        mae_list, mse_list, rae_list, r2_list = [], [], [], []
        for train_idx, test_idx in kf.split(X_topk):
            X_train, y_train = X_topk.iloc[train_idx, :], y[train_idx]
            X_test, y_test = X_topk.iloc[test_idx, :], y[test_idx]
            
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            
            mae_list.append(1 - mean_absolute_error(y_test, y_pred))
            mse_list.append(1 - mean_squared_error(y_test, y_pred))
            rae_list.append(1 - relative_absolute_error(y_test, y_pred))
            r2_list.append(r2_score(y_test, y_pred))
        
        metrics = {
            'mean_MAE': np.mean(mae_list),
            'mean_MSE': np.mean(mse_list),
            'mean_RAE': np.mean(rae_list),
            'mean_R2':  np.mean(r2_list)
        }
    
    return list(top_k_features), metrics

def report_performance(X,y, task_type='cls'):
    X = X
    y = y.astype(int) 
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list = [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average=
            'weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted')
                            )
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list = [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list)
    else:
        return -1


