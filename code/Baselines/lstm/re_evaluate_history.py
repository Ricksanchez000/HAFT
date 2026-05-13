import argparse
import pickle
import sys

import pandas
import os

import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('./')

from utils.datacollection.logger import info, error
from Record import TransformationRecordList, TransformationRecord
from feature_env import TransformationFeatureEvaluator
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--task_name', type=str, choices=['airfoil', 'amazon_employee',
                                                          'ap_omentum_ovary', 'german_credit',
                                                          'higgs', 'housing_boston', 'ionosphere',
                                                          'lymphography', 'messidor_features', 'openml_620',
                                                          'pima_indian', 'spam_base', 'spectf', 'svmguide3',
                                                          'uci_credit_card', 'wine_red', 'wine_white', 'openml_586',
                                                          'openml_589', 'openml_607', 'openml_616', 'openml_618',
                                                          'openml_637'], default='airfoil')
args = parser.parse_args()
fe = TransformationFeatureEvaluator(args.task_name, from_scratch=False)


records = list(fe.records.r_list)
neo_rl_list = TransformationRecordList(fe.records.max_size)

max_p = -1
best_trans: TransformationRecord = None
max_train_p = -1
performances = []
for i in tqdm.tqdm(records):
    i.valid = True
    try:
        test_performance = fe.get_performance(i.op(fe.test))
        train_performance = fe.get_performance(i.op(fe.train))
        performances.append((test_performance, train_performance))
        neo_rl_list.append(i.operation, test_performance)
        if test_performance > max_p:
            best_trans = i
            max_p = test_performance
        if train_performance > max_train_p:
            max_train_p = train_performance
    except:
        error(f'error happend with {i.operation}, continue!')
info(f'the best test performance for {args.task_name} is : {max_p}')
info(f'the best train performance for {args.task_name} is : {max_train_p}')
info(f'proceed {len(neo_rl_list)} stuff, the best is {best_trans.operation}, its performance is {max_p}')
base_path = './data'

suppose_file_cache_name = f'{base_path}/history/{args.task_name}/fe.pkl'
with open(suppose_file_cache_name, 'wb') as f:
    pickle.dump(neo_rl_list, f)

suppose_performance_name = f'{base_path}/history/{args.task_name}/performances.tuple.pkl'
with open(suppose_performance_name, 'wb') as f:
    pickle.dump(performances, f)