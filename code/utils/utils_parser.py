# utils_parser.py
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=[
        'airfoil', 'amazon_employee', 'ap_omentum_ovary', 'german_credit',
        'higgs', 'housing_boston', 'ionosphere', 'lymphography',
        'messidor_features', 'openml_620', 'pima_indian', 'spam_base',
        'spectf', 'svmguide3', 'uci_credit_card', 'wine_red', 'wine_white',
        'openml_586', 'openml_589', 'openml_607', 'openml_616', 'openml_618',
        'openml_637'], default='wine_white')
    return parser



def para_parser():
    parser = argparse.ArgumentParser(description="Feature Generation Experiment")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to be processed.")
    parser.add_argument('--id', type=str, default='0', help='give this exp a special id!')
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--num_episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--critic_lr", type=float, default=1e-4, help="Critic Learning rate")
    parser.add_argument("--feature_lr", type=float, default=1e-4, help="Feature Agent Learning rate")
    parser.add_argument("--op_coef", type=float, default=0.1, help="Actor Learning rate")
    parser.add_argument("--ent_coef", type=float, default=0.1, help="Actor Learning rate")
    parser.add_argument("--n_splits", type=int, default=2, help="n_splits for cross validation")
    parser.add_argument("--num_layer", type=int, default=2, help="n_layer for transformation encoder")
    parser.add_argument("--top_k", type=int, default=20, help="top k features to keep for AP")
    # Only used for dataset 'ap_omentum_ovary': number of features kept after initial screening
    parser.add_argument("--ap_k", type=int, default=100, help="Initial SelectKBest k for ap_omentum_ovary")
    
    return parser
