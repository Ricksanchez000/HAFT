# main.py

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import argparse
import random

seed_value = 0
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

from collections import namedtuple, defaultdict
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler
from new2_environment_final import FeatureGenerationEnv,TASK_DICT
from new2_feature_agent_final import PPOAgent
from new2_operation_agent_final_PPO import OperationAgentPPO
from shared_critic import SharedCritic, FeatureSetAttention
from memory import Memory,OperationMemory,Memory_happo
from utils.utils_parser import *
from utils.logger import *
from utils.tools_final import generate_action_mask,evaluate_features,test_task_new,report_performance
from utils.tools_final import select_topk_features_and_evaluate,compute_relevance_Rv,compute_redundancy_Rd
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
from utils.tools_final import calculate_meta_statistics,compute_gae,calculate_meta_statistics_with_attention


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


O1 = ['sqrt', 'square', 'cos', 'sin', 'tanh', 
      'cube', 'exp', 'sigmoid', 'log', 'reciprocal']
O2 = ['+', '-', '*', '/']
O3 = ['minmax_scaler', 'quan_trans']

operation_list = O1 + O2 + O3
one_hot_op = pd.get_dummies(operation_list)
operation_emb = defaultdict()
for item in one_hot_op.columns:
    operation_emb[item] = torch.tensor(one_hot_op[item].values, dtype=torch.float32)
print(operation_emb)
OP_DIM = len(operation_list)
print(OP_DIM)


def main(params):

    total_start_time = time.time()

    num_episodes = params['num_episodes']
    max_steps_per_episode = params['max_steps']
    critic_lr = params['critic_lr'] 
    op_coef = params['op_coef']
    feature_lr = params['feature_lr']
    n_splits = params['n_splits']
    ent_coef = params['ent_coef']
    top_k = int(params.get('top_k', 100))

    max_features = 85 
    feature_dim = 7   

    dataset_name = params['dataset']
    task_type = TASK_DICT[dataset_name]
    print(f"Dataset Name: {dataset_name}; task type: {task_type}")
    base_path = os.path.abspath("Test2")
    data_path = os.path.join(base_path, f"{dataset_name}.hdf")
    print(f"Data path: {data_path}")
    try:
        data = pd.read_hdf(data_path, 'wdj')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        exit()
    except ValueError as e:
        print(f"Error loading HDF5 file: {e}")
        exit()
         
    #Data preprocessing
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    if task_type == 'cls':
        y = y.astype(int)
    if dataset_name in ('ionosphere', 'ap_omentum_ovary','openml_586'):
        X = pd.DataFrame(X.values, columns=X.columns).astype(float)
        print(f"[Preprocess] Skipping MinMaxScaler for {dataset_name}")
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
    X.columns = X.columns.astype(str)
    # following previous work
    # 80% data for generate new feature
    # 20% data for iteratively evaluate the performance of generated features and select the best feature set, assuming IID data distribution as heuristic search
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0, shuffle=True) 
    if dataset_name == 'ap_omentum_ovary':
        k = min(int(params.get('ap_k', 200)), X_train.shape[1])
        selector = SelectKBest(mutual_info_regression, k=k).fit(X_train, y_train)
        cols = selector.get_support()
        X_train = X_train.loc[:, cols]
        X_test = X_test.loc[:, cols]
        print(f"[Preprocess] AP SelectKBest fit on train split: k={k}")
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    feature_df = X_train
    target_series = y_train

    #info(f'check the tyep of y: {y.unique()}')
    #info(f'check the tyep of y: {y.dtype}')
    #initialize the environment
    env = FeatureGenerationEnv(feature_df, target_series,X_test,
                               max_features=max_features,task_name = dataset_name)
    
    X = feature_df[env.original_features]
    if task_type == 'reg':
        original_MAE, original_MSE, original_RAE  = report_performance(X_test,y_test,task_type=task_type)
        print(f'Original 1-MAE: {original_MAE}')
        print(f'Original 1-MSE: {original_MSE}')
        print(f'Original 1-RAE: {original_RAE}')
    elif task_type == 'cls':
        original_pre,original_rec, original_f1 = report_performance(X_test,y_test,task_type=task_type)
        print(f'Original pre: {original_pre}')
        print(f'Original rec: {original_rec}')
        print(f'Original F1: {original_f1}')
    
    #initialize the framework components
    z_dim = 10
    critic_input_dim = feature_dim * feature_dim + z_dim
    #critic_input_dim = feature_dim * feature_dim     
    
    shared_critic = SharedCritic(input_dim=critic_input_dim)   
    #critic_optimizer = optim.Adam(shared_critic.parameters(), lr=critic_lr) 
    
    
    feature_attention = FeatureSetAttention(feature_dim=feature_dim,embed_dim=64,num_heads=4,hidden_dim=z_dim)
    all_critic_params = list(shared_critic.parameters()) + list(feature_attention.parameters())
    critic_optimizer = optim.Adam(all_critic_params, lr=critic_lr)
    

    feature_agent_head = PPOAgent(feature_dim=feature_dim,select='head',
                             shared_critic=shared_critic,feature_lr=feature_lr)      
    operation_agent = OperationAgentPPO(input_dim=feature_dim+OP_DIM,
                               action_dim=OP_DIM,
                               shared_critic=shared_critic)
    feature_agent_tail = PPOAgent(feature_dim=feature_dim,select='tail',
                             shared_critic=shared_critic,feature_lr=feature_lr,operation_dim = OP_DIM,operation_emb = operation_emb,)
    
    best_mse = float('inf')
    best_1rae = -float('inf') 
    best_cross_entropy = float('inf')
    best_f1 = float('0')
    
    best_RAE = -float('inf')
    best_F1 = float('0')

    best_features = env.current_features.copy()
    episode_records = []
    
 

    time_step_list = []
    #reward_calculation_time = []
    #time_action_list = []
    #feature_selection_time_list = []

    for episode in range(num_episodes):

        state = env.reset()          
                           
        feature_memory1 = Memory_happo()
        feature_memory2 = Memory_happo()  
        operation_memory = OperationMemory() 

        global_rewards = []
        global_critic_states = []
        global_dones = []

   

        training_start_time = time.time()
        for step in range(max_steps_per_episode):
            info(f'episode:{episode+1} | step:{step+1}')
            info('-----------------')
            feature_names = state['features']    
            feature_data = state['feature_data']  

            step_start_time = time.time()

            feature_data_tensor = torch.tensor(feature_data,
                                               dtype=torch.float32).unsqueeze(0)  
            start_time_action = time.time()
            #head_start_time = time.time()
            action_f1, logprob_f1 = feature_agent_head._select_head(
                feature_data_tensor) 
            
            #info("=="*40)
            #info(f"check the action_f1:{action_f1}")
            #info("=="*40)
            #info(f"check the logprob_f1:{logprob_f1}")


            #head_end_time = time.time()
            #head_selection_time = head_end_time - head_start_time
            #transformer_time = head_selection_time*2
            #transformer_running_time.append(transformer_time)

            feature1 = feature_names[action_f1]     #feature1 str
            feature1_data = feature_data[action_f1] #feature1 REP

            op_state = feature1_data  #REP(feature1) as op_state
            
            feature1_data_real = env.feature_df[feature1].to_numpy() 

            action_op,logprob_op,entropy_op = operation_agent.select_action(op_state,op_coef=op_coef,action_mask=None) 
           
            operation = operation_list[action_op]

            # Initialize action_f2 and logprob_f2 to None，might use it in later memory record
            action_f2 = None
            logprob_f2 = None
            
            # Define the operation lists
            unary_operations = O1 + O3
            binary_operations = O2



            mask = generate_action_mask(feature1_data_real, operation_list)

           # Initial operation selection
            action_op, logprob_op,entropy_op = operation_agent.select_action(op_state,op_coef=op_coef,action_mask=mask)
            operation = operation_list[action_op]

            # Proceed directly with the selected operation
            if operation in unary_operations:
                info(f'Unary op selected')
                info(f'Selected op: {operation},    h_f: {feature1}')
                info(f'(Generated Feature: {feature1}_{operation})')
                end_time_action = time.time()

                #reward_cal_start_time = time.time()
                next_state, reward, mse, done = env.step(action_f1, None, action_op)
                #reward_cal_end_time = time.time()
                info(f'step:{step+1} and immediate reward:{reward}')
            elif operation in binary_operations:
                info(f'Binary op selected')
                while True:
                    action_f2, logprob_f2 = feature_agent_tail._select_tail(
                        feature_data_tensor, feature1_data, op=operation)
                    if action_f2 != action_f1:
                        break
                feature2 = feature_names[action_f2]
                end_time_action = time.time()

                #reward_cal_start_time = time.time()
                if task_type == 'reg':
                    next_state, reward, rae1, done = env.step(action_f1, action_f2, action_op=action_op) 
                elif task_type == 'cls':
                    next_state, reward, cross_entropy, done = env.step(action_f1, action_f2, action_op=action_op)
                #reward_cal_end_time = time.time()


                info(f'Selected op: {operation},    h_f:{feature1},     t_f:{feature2}')
                info(f'(Generated Feature: {feature1}_{operation}_{feature2})')
                info(f'step:{step+1} and immediate reward:{reward}')
            else:
                info('Unrecognized operation encountered, skipping.')
                pass
            
            #reward_cal_step_time = reward_cal_end_time - reward_cal_start_time
            #reward_calculation_time.append(reward_cal_step_time)
            #info(f'reward_cal_step_time: {reward_cal_step_time}')

            #time_action = end_time_action - start_time_action
            info(f'time_action this step: {time}')
            #time_action_list.append(time_action)

            #X_for_mi = env.feature_df[env.current_features]
            #a = 1e-5
            #b = 1e-5
            #Rv = a*compute_relevance_Rv(X_for_mi, y)
            #Rd = b*compute_redundancy_Rd(X_for_mi)
            #info(f'Rc = : {Rv}')
            #info(f'Rd = : {Rd}')


            feature_memory1.obs.append(feature_data)
            feature_memory1.actions.append(torch.tensor(action_f1))
            feature_memory1.logprobs.append(logprob_f1)
            feature_memory1.mask.append(1.0)

            feature_memory2.obs.append(feature_data)
            if action_f2 is None:
                feature_memory2.actions.append(torch.tensor(0))          
                feature_memory2.logprobs.append(torch.tensor(0.0))                          
                feature_memory2.mask.append(0.0)                          
            else:
                feature_memory2.actions.append(torch.tensor(action_f2))
                feature_memory2.logprobs.append(logprob_f2)
                feature_memory2.mask.append(1.0)

            #  meta_state
            #info(f"check feature_data shape: {feature_data.shape}")
            
            meta_state = calculate_meta_statistics_with_attention(feature_data, feature_attention)
            meta_state_t = torch.tensor(meta_state, dtype=torch.float32)
            '''
            meta_state = calculate_meta_statistics(feature_data).flatten()
            meta_state_t = torch.tensor(meta_state, dtype=torch.float32)
            '''
            #info(f"check meta_state_t shape: {meta_state_t.shape}")
            global_rewards.append(reward)
            global_critic_states.append(meta_state_t)
            global_dones.append(done)

            # operation agent
            op_input = np.concatenate([op_state, mask])  
            #operation_memory.states.append(op_input)
            operation_memory.obs.append(torch.as_tensor(op_input, dtype=torch.float32))
            operation_memory.actions.append(torch.tensor(action_op))
            operation_memory.logprobs.append(logprob_op)
            operation_memory.mask.append(1.0)                
            


            
            state = next_state 

            step_end_time = time.time()
            total_step_time = step_end_time - step_start_time
            time_step_list.append(total_step_time)
            #info('#############report 3 function block time consumption per step###############')
            #info(f'Agent action time of this step: {time_action}')
            #info(f'reward calculation time of this step : {reward_cal_step_time}')
            #info(f'Total using time for step-{step+1}: {total_step_time}s')


            if done:
                break
                
                

        info(f"max(head_actions)={max(int(x) for x in feature_memory1.actions)}")
        info(f"max(tail_actions)={max(int(x) for x in feature_memory2.actions)}")
        '''Calculate meta state for shared critic'''
        # values: critic
        
        values = shared_critic(torch.stack(global_critic_states)).squeeze(-1)      # [T]
        '''
        bootstrap_v = 0.0 if global_dones[-1] else shared_critic(
            torch.tensor(calculate_meta_statistics(next_state['feature_data']).flatten(),
                        dtype=torch.float32).unsqueeze(0)
        ).item()

        '''
        bootstrap_v = 0.0 if global_dones[-1] else shared_critic(
            torch.tensor(
                calculate_meta_statistics_with_attention(
                    next_state['feature_data'], 
                    feature_attention
                ),
                dtype=torch.float32
            ).unsqueeze(0)
        ).item()
        
        
        values = torch.cat([values, torch.tensor([bootstrap_v])])  # [T+1]

        # 建议：
        with torch.no_grad():
            returns, adv = compute_gae(torch.tensor(global_rewards, dtype=torch.float32),
                                values, gamma=0.99, lam=0.95)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.detach()                     
        info(f'✅gae_returns: {returns}')



        agents_and_mems = [
            ("head", feature_agent_head, feature_memory1),
            ("op",   operation_agent,   operation_memory), 
            ("tail", feature_agent_tail, feature_memory2),
        ]
        factor = torch.ones_like(adv)
        ent_coef = 0.01

        #critic update
        critic_optimizer.zero_grad()
        pred_v = values[:-1]  # [T]
        value_loss = 0.5 * torch.mean((pred_v - returns)**2)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(shared_critic.parameters(), 0.5)
        critic_optimizer.step()


        '''
        for name, agent, mem in agents_and_mems:
            for i, s in enumerate(mem.obs):
                print(f"State {i} shape ‼️: {np.array(s).shape}")
            
            for i, s in enumerate(mem.actions):
                print(f"Action {i} shape ‼️: {np.array(s).shape}")

            if len(mem.logprobs) == 0:
                continue
            pol_loss, factor = agent.happo_step_iter(
                mem,             # memory（里面有 obs/actions/logprobs/mask）
                adv, factor,     # 全局 advantage & 当前 factor
                clip_eps=agent.eps_clip, ent_coef = ent_coef)
        '''

        #Plain shared critic update
        
        '''
        total_policy_loss,total_value_loss = feature_agent_head.update(feature_memory1,
                             critic_optimizer, critic_input_dim)  
        if len(feature_memory2.logprobs) > 0:
            total_policy_loss,total_value_loss = feature_agent_tail.update(feature_memory2,
                             critic_optimizer, critic_input_dim)
        else:
            pass
        
        feature_memory1.clear_memory() 
        feature_memory2.clear_memory() 
        
        
        operation_agent.update(operation_rewards, operation_log_probs,
                               operation_states, critic_optimizer,
                               critic_input_dim,operation_entropy)
        
        '''


        
        if task_type == 'reg':
            mse, rae = evaluate_features(env, env.current_features, task_type='reg',n_splits=n_splits)
            print(f'Episode {episode+1}, MSE: {mse}, 1-RAE:{rae}')
        else:
            f1, cross_entropy = evaluate_features(
                env,
                env.current_features,
                task_type='cls',
                n_splits=n_splits,
                allow_log_loss_fallback=(dataset_name == 'lymphography')
            )
            print(f'Episode {episode+1}, F1: {f1}, Cross Entropy:{cross_entropy}')


        #top_k = 20
        #top_k = 50
        #top_k = 100, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20       
        #best_top_K_features, metrics = select_topk_features_and_evaluate(env.feature_df, y, top_k, task_type, n_splits=n_splits)
        best_top_K_features, metrics = select_topk_features_and_evaluate(env.feature_df_test, y_test, top_k, task_type, n_splits=n_splits)

        if task_type == 'reg':
            current_RAE = metrics['mean_RAE']
            if current_RAE > best_RAE:
                best_features_global = best_top_K_features.copy()
                best_MAE = metrics['mean_MAE']
                best_MSE = metrics['mean_MSE']
                best_R2  = metrics['mean_R2']
                best_RAE = current_RAE
            episode_records.append({
            'episode': episode + 1,
            'current_reward': current_RAE,
            'best_reward': best_RAE
            })
        elif task_type == 'cls':
            current_F1 = metrics['mean_f1']
            if  current_F1 > best_F1:
                best_features_global = best_top_K_features.copy()
                best_pre = metrics['mean_precision']
                best_rec = metrics['mean_recall']
                best_acc = metrics['mean_accuracy']
                best_F1 = current_F1
            episode_records.append({
            'episode': episode + 1,
            'current_reward': current_F1,
            'best_reward': best_F1
            })


        if task_type == 'reg':
            info(f"episode {episode + 1} ; top_k features: {best_top_K_features} ; current best Metrics: Best 1-MAE {best_MAE}; Best 1 - MSE {best_MSE}; Best 1-RAE {best_RAE}")
        elif task_type == 'cls':
            info(f"episode {episode + 1} ; top_k features: {best_top_K_features} ; current best Metrics: Best Pre {best_pre}; Best rec {best_rec}; Best F1 {best_F1}")
        info(f'Total using time for episode {episode+1}: {time.time() - training_start_time:.1f}s')

    if task_type == 'reg':
        print(f'Original 1-MAE: {original_MAE} | Original 1-MSE: {original_MSE} | Original 1-RAE: {original_RAE}')
        metric_suffix = f"_MSE{best_MSE:.4f}_1RAE{best_RAE:.4f}"
        info(f'Global best feature set: {best_features_global}; Global best metrics: best 1-MAE:{best_MAE}; best 1-MSE:{best_MSE}; best 1-RAE:{best_RAE}; best r2:{best_R2} ')
    elif task_type == 'cls':
        print(f'Original pre: {original_pre} | Original rec: {original_rec} | Original F1: {original_f1}')
        metric_suffix = f"_F1{best_F1:.4f}_CE{best_cross_entropy:.4f}"
        info(f'Global best feature set: {best_features_global}; Global best metrics: best pre: {best_pre}; best rec: {best_rec}; best F1: {best_F1}; best acc:{best_acc} ')
    else:
        metric_suffix = ""

    print('Best feature set:')
    print(best_features)
    

    best_feature_df = env.feature_df[best_features_global].copy()
    best_feature_df["label"] = env.target
    best_feature_test_df = env.feature_df_test[best_features_global].copy()
    best_feature_test_df["label"] = y_test

    result_dir = params['result_dir']
    dataset_name = params['dataset']
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename_csv = os.path.join(result_dir, f"{dataset_name}_{timestamp}_best_features_{metric_suffix}.csv")
    filename_test_csv = os.path.join(result_dir, f"{dataset_name}_{timestamp}_best_features_test_{metric_suffix}.csv")
    best_feature_df.to_csv(filename_csv, index=False)
    best_feature_test_df.to_csv(filename_test_csv, index=False)
    dataset_dir = os.path.dirname(filename_csv)
    records_filename = os.path.join(dataset_dir, f"{dataset_name}_{timestamp}_episode_rewards.csv")
    records_df = pd.DataFrame(episode_records)
    records_df.to_csv(records_filename, index=False)
    print(f"Best train feature DataFrame (with label) saved to {filename_csv}")
    print(f"Best test feature DataFrame (with label) saved to {filename_test_csv}")
    print(f"Episode rewards DataFrame saved to {records_filename}")

    logger.info(f"Dataset {dataset_name} completed successfully. Train results saved to {filename_csv}; test results saved to {filename_test_csv}")
    print(f"Dataset {dataset_name} completed successfully. Train results saved to {filename_csv}; test results saved to {filename_test_csv}")
    
    # End time of all episodes 
    total_end_time = time.time()  
    total_time_spent = total_end_time - total_start_time  
    info(f'Total time spent for all episodes: {total_time_spent:.1f}s')  

if __name__ == '__main__':

    parser = para_parser()
    args = parser.parse_args()
    params = vars(args)

    trail_id = params.get('id', 'default')  
    dataset_name = params['dataset']
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    base_log_dir = './log2/'
    base_result_dir = './results5/'

    dataset_log_dir = os.path.join(base_log_dir, trail_id, dataset_name)
    dataset_result_dir = os.path.join(base_result_dir, trail_id, dataset_name)
    
    os.makedirs(dataset_log_dir, exist_ok=True)
    os.makedirs(dataset_result_dir, exist_ok=True)

    log_file = os.path.join(dataset_log_dir, f"{timestamp}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s : %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    logger = logging.getLogger('')
    
    logger.info(f"Starting experiment for dataset: {dataset_name}")
    logger.info(f"Parameters: {params}")

    params['log_dir'] = dataset_log_dir
    params['result_dir'] = dataset_result_dir

    main(params)

    print("\n===== Hyperparameters Used =====")
    for k, v in params.items():
        print(f"{k}: {v}")
 
    logger.info("===== Hyperparameters Used at End =====")
    logger.info(params)
