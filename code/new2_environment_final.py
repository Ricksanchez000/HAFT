# environment.py

import random
import pandas as pd
import numpy as np
import torch

seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, f1_score, log_loss
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer


from utils.reward import relative_absolute_error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#DICT for task name and task type
TASK_DICT = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls','credit_default': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg'
             }

class FeatureGenerationEnv:
    def __init__(self, feature_df, target_series,feature_df_test, max_features=20, task_type = None,task_name = None):
        self.original_features = feature_df.columns.tolist() 
        self.feature_df = feature_df.copy()
        self.target = target_series
        self.task_name = task_name
        self.current_features = self.original_features.copy()
        
        self.feature_df_test = feature_df_test.copy()
        self.original_features_test = feature_df_test.columns.tolist()
        self.test_features = self.original_features_test.copy() 
        
        self.max_features = max_features   
        self.reset()

        if task_type is None:
            self.task_type = TASK_DICT[task_name] 
        else:
            self.task_type = task_type
        

    def reset(self):
        self.current_features = self.original_features.copy()
        self.update_state()
        return self.state
    
    def update_state(self):
        self.state = {
            'features': self.current_features.copy(),                    
            'feature_data': self.get_feature_data(self.current_features) 
        }

    def get_feature_data(self, feature_names):
        feature_data = []
        for f in feature_names:
            feature_series = self.feature_df[f].astype(np.float64)   
            stats = feature_series.describe().fillna(0)
            stats = stats.drop('count') 
            data = stats.tolist()
            feature_data.append(data)
        feature_data = np.array(feature_data)
        return feature_data

    def step(self, action_f1, action_f2=None, action_op=None):
        
        """
        Execute one step of the environment given the action choices.
        Args:
            action_f1 (int): Index of the first feature (in current_features).
            action_f2 (int, optional): Index of the second feature. Defaults to None for unary operations.
            action_op (int): Index of the selected operation in operation_list.
        Returns:
            (state, reward, metric, done): Next state, reward, metric (MSE or F1, etc.), and a boolean indicating if done.
        """
        feature1 = self.current_features[action_f1] 

        O1 = ['sqrt', 'square', 'cos', 'sin', 'tanh', 
            'cube', 'exp', 'sigmoid', 'log', 'reciprocal']
        O2 = ['+', '-', '*', '/']
        O3 = ['minmax_scaler', 'quan_trans']
        
        operation_list = O1 + O2 + O3
        operation = operation_list[action_op]   

        if action_f2 == None:
            new_feature_name = f'({feature1}_{operation}_{feature1})'
            if new_feature_name not in self.feature_df.columns:
                self.generate_new_feature(f1=feature1,
                                          operation=operation, new_feature_name=new_feature_name)
                self.current_features.append(new_feature_name)
                self.test_features.append(new_feature_name) 
        else:
            feature2 = self.current_features[action_f2]
            new_feature_name = f'({feature1}_{operation}_{feature2})'
            if new_feature_name not in self.feature_df.columns:
                self.generate_new_feature(feature1, feature2,
                                          operation=operation, new_feature_name=new_feature_name)
                self.current_features.append(new_feature_name)
                self.test_features.append(new_feature_name)

        reward,mse = self.evaluate_model(task_type=self.task_type)
        self.update_state()
        done = False
        return self.state, reward, mse, done
    
    def generate_new_feature(self, f1, f2=None, operation=None, new_feature_name=None):
        #Binary Operations
        if operation == '+':
            self.feature_df[new_feature_name] = \
                self.feature_df[f1] + self.feature_df[f2]
            self.feature_df_test[new_feature_name] = \
                self.feature_df_test[f1] + self.feature_df_test[f2]
        elif operation == '-':
            self.feature_df[new_feature_name] = \
                self.feature_df[f1] - self.feature_df[f2]
            self.feature_df_test[new_feature_name] = \
                self.feature_df_test[f1] - self.feature_df_test[f2]
        elif operation == '*':
            self.feature_df[new_feature_name] = \
                self.feature_df[f1] * self.feature_df[f2]
            self.feature_df_test[new_feature_name] = \
                self.feature_df_test[f1] * self.feature_df_test[f2]
        elif operation == '/': 
            self.feature_df[new_feature_name] = \
                self.feature_df[f1] / (self.feature_df[f2] + 1e-5) 
            self.feature_df_test[new_feature_name] = \
                self.feature_df_test[f1] / (self.feature_df_test[f2] + 1e-5) 
        #Unary Operations
        elif operation == 'sin':
            self.feature_df[new_feature_name] = \
                np.sin(self.feature_df[f1])
            self.feature_df_test[new_feature_name] = \
                np.sin(self.feature_df_test[f1])
        elif operation == 'cos':
            self.feature_df[new_feature_name] = \
                np.cos(self.feature_df[f1])
            self.feature_df_test[new_feature_name] = \
                np.cos(self.feature_df_test[f1])
        elif operation == 'tanh':
            self.feature_df[new_feature_name] = \
                np.tanh(self.feature_df[f1])
            self.feature_df_test[new_feature_name] = \
                np.tanh(self.feature_df_test[f1])
        elif operation == 'sigmoid':
            from scipy.special import expit
            self.feature_df[new_feature_name] = \
                expit(self.feature_df[f1])  
            self.feature_df_test[new_feature_name] = \
                expit(self.feature_df_test[f1])  
        elif operation == 'cube':
            def cube(x):
                return x ** 3
            self.feature_df[new_feature_name] = \
                cube(self.feature_df[f1]) 
            self.feature_df_test[new_feature_name] = \
                cube(self.feature_df_test[f1])  
        elif operation == 'square':
            self.feature_df[new_feature_name] = \
                np.square(self.feature_df[f1])
            self.feature_df_test[new_feature_name] = \
                np.square(self.feature_df_test[f1])
        elif operation == 'exp':
            f1_clipped = np.clip(self.feature_df[f1], None, 11)
            self.feature_df[new_feature_name] = np.exp(f1_clipped)
            self.feature_df_test[new_feature_name] = np.exp(f1_clipped)
        elif operation == 'sqrt':
            self.feature_df[new_feature_name] = \
                np.sqrt(self.feature_df[f1] + 1e-5)      
            self.feature_df_test[new_feature_name] = \
                np.sqrt(self.feature_df_test[f1] + 1e-5)  
        elif operation == 'log':
                self.feature_df[new_feature_name] = \
                    np.log(self.feature_df[f1] + 1e-5)  
                self.feature_df_test[new_feature_name] = \
                    np.log(self.feature_df_test[f1] + 1e-5)  
        elif operation == 'reciprocal':
            self.feature_df[new_feature_name] = \
                1 / (self.feature_df[f1] + 1e-5)  
            self.feature_df_test[new_feature_name] = \
                1 / (self.feature_df_test[f1] + 1e-5)  
        elif operation == 'stand_scaler':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.feature_df[new_feature_name] = \
                scaler.fit_transform(self.feature_df[[f1]])
            scaler = StandardScaler()
            self.feature_df_test[new_feature_name] = \
                scaler.fit_transform(self.feature_df_test[[f1]])
        elif operation == 'minmax_scaler':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(-1, 1))
            self.feature_df[new_feature_name] = \
                scaler.fit_transform(self.feature_df[[f1]])
            scaler = MinMaxScaler(feature_range=(-1, 1))
            self.feature_df_test[new_feature_name] = \
                scaler.fit_transform(self.feature_df_test[[f1]])
        elif operation == 'quan_trans':
            from sklearn.preprocessing import QuantileTransformer
            transformer = QuantileTransformer(random_state=0)
            self.feature_df[new_feature_name] = \
                transformer.fit_transform(self.feature_df[[f1]])
            transformer = QuantileTransformer(random_state=0)
            self.feature_df_test[new_feature_name] = \
                transformer.fit_transform(self.feature_df_test[[f1]])
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def evaluate_model(self,task_type = None,n_splits = 2):
        
        X = self.feature_df[self.current_features]
        y = self.target

        if task_type == 'reg':
            kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
            reg = RandomForestRegressor(random_state=0) 
            rae_list = []
            mse_list = []
            for train, test in kf.split(X):
                X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
                ], X.iloc[test, :], y.iloc[test]
                reg.fit(X_train, y_train)
                y_predict = reg.predict(X_test)
                rae_list.append(1 - relative_absolute_error(y_test, y_predict))
                mse_list.append(mean_squared_error(y_test, y_predict))
            return np.mean(mse_list),np.mean(rae_list) 
    
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
                try:
                    cross_entropy_list.append(log_loss(y_test, y_proba))
                except ValueError:
                    if self.task_name != 'lymphography':
                        raise
                    labels = clf.classes_
                    mask = np.isin(y_test, labels)
                    if not np.any(mask):
                        continue
                    cross_entropy_list.append(log_loss(y_test[mask], y_proba[mask], labels=labels))
            
            return np.mean(f1_list), np.mean(cross_entropy_list)
        else:
            return -1
        
