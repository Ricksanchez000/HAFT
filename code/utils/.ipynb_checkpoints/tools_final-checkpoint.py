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
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error)
import warnings
warnings.filterwarnings("ignore", module=r"sklearn(\.|$)")


seed_value = 0
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

def cube(x):
    return x ** 3
def generate_action_mask(feature_data: np.ndarray, operation_list: list) -> list:
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
            np.mean(col_data),       
            np.std(col_data),         
            np.min(col_data),         
            np.max(col_data),         
            np.percentile(col_data, 25),  
            np.percentile(col_data, 50),  
            np.percentile(col_data, 75)   
        ]
        meta_stats.append(meta_statistics)
    meta_stats = np.array(meta_stats)  
    flat = meta_stats.flatten()
    val_min = flat.min()
    val_max = flat.max()

    EPS = 1e-8
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
        emb_vec = operation_emb[op_name]  
        op_tensors.append(emb_vec.unsqueeze(0))  
    op_seq = torch.cat(op_tensors, dim=0)
    return op_seq

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

def report_performance(X,y, task_type='cls',n_splits = 5):
    X = X
    y = y.astype(int) 
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


def compute_relevance_Rv(X: pd.DataFrame, y: pd.Series, task_type='cls'):
    """
    Rv = (1/|N|) * Σ I(x_i, y)
    """
    if task_type == 'cls':
        mi_values = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
    else:
        mi_values = mutual_info_regression(X, y, discrete_features='auto', random_state=0)
    Rv = np.mean(mi_values)
    return Rv

def compute_redundancy_Rd(X: pd.DataFrame, bins=10):
    """
    Rd = (1/|N|^2) * Σ I(x_i, x_j)
    """
    X_binned = pd.DataFrame()
    for col in X.columns:
        X_binned[col] = pd.cut(X[col], bins=bins, labels=False, include_lowest=True)
    
    n = X_binned.shape[1]
    sum_mi = 0.0
    for i in range(n):
        for j in range(n):
            mi_ij = mutual_info_score(X_binned.iloc[:, i], X_binned.iloc[:, j])
            sum_mi += mi_ij
    
    Rd = sum_mi / (n**2)
    return Rd


def compute_gae(rewards, values, gamma=0.99, lam=0.95):

    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32)
    
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    
    returns = advantages + values[:-1]
    return returns, advantages


def calculate_meta_statistics_with_attention(feature_data, attention_module=None):
    """
    Combines statistical summary with attention-based interactions.
    
    Args:
        feature_data: numpy array of shape [num_features, feature_dim]
        attention_module: Instance of FeatureSetAttention
    
    Returns:
        combined_state: Concatenated statistics and interaction vector
    """
    meta_state = calculate_meta_statistics(feature_data).flatten()  # [49]
    
    if attention_module is not None:
        feature_tensor = torch.tensor(feature_data, dtype=torch.float32)
        with torch.no_grad():
            interaction_vector = attention_module(feature_tensor)  # [hidden_dim]
        interaction_vector = interaction_vector.numpy()
        combined_state = np.concatenate([meta_state, interaction_vector])
    else:
        combined_state = meta_state
    
    return combined_state



def select_topk_features_and_evaluate_new(
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
