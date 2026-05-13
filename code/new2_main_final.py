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
import psutil  # Add this for memory tracking
import tracemalloc  # Add this for detailed memory tracking

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

def get_memory_info():
    """Get current memory usage information"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # Get GPU memory if available
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    
    return {
        'rss_gb': mem_info.rss / 1024**3,  # Resident Set Size in GB
        'vms_gb': mem_info.vms / 1024**3,  # Virtual Memory Size in GB
        'percent': process.memory_percent(),  # Percentage of system memory
        'gpu_gb': gpu_memory  # GPU memory in GB
    }


def main(params):

    total_start_time = time.time()
    
    # Start memory tracking
    tracemalloc.start()
    initial_memory = get_memory_info()

    # Seed handling
    seed = int(params.get('seed', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    num_episodes = params['num_episodes']
    max_steps_per_episode = params['max_steps']
    critic_lr = params['critic_lr'] 
    op_coef = params['op_coef']
    feature_lr = params['feature_lr']
    n_splits = params['n_splits']
    ent_coef = params['ent_coef']

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
         
    if params.get('dataset') == 'ap_omentum_ovary':
        k = int(params.get('ap_k', 100))
        selector = SelectKBest(mutual_info_regression, k=k).fit(data.iloc[:, :-1], data.iloc[:, -1])
        cols = selector.get_support()
        X_new = data.iloc[:, :-1].loc[:, cols]
        data = pd.concat([X_new, data.iloc[:, -1].astype(int)], axis=1)

    #Data preprocessing
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)  
    X.columns = X.columns.astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True
    ) 
    X_train = X_train.reset_index(drop=True) # feature generation  set
    X_test = X_test.reset_index(drop=True)   # downstream task iterative optimization set, IID assumed as heuristic search
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
        original_MAE, original_MSE, original_RAE  = report_performance(X_test,y_test,task_type=task_type,n_splits=n_splits)
        print(f'Original 1-MAE: {original_MAE}')
        print(f'Original 1-MSE: {original_MSE}')
        print(f'Original 1-RAE: {original_RAE}')
    elif task_type == 'cls':
        original_pre,original_rec, original_f1 = report_performance(X_test,y_test,task_type=task_type,n_splits=n_splits)
        print(f'Original pre: {original_pre}')
        print(f'Original rec: {original_rec}')
        print(f'Original F1: {original_f1}')


    
    z_dim = 10
    critic_input_dim = feature_dim * feature_dim + z_dim
    #critic_input_dim = feature_dim * feature_dim     
    
    shared_critic = SharedCritic(input_dim=critic_input_dim)   
    #ritic_optimizer = optim.Adam(shared_critic.parameters(), lr=critic_lr) 
    
    
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

    best_MAE = np.nan
    best_MSE = np.nan
    best_R2  = np.nan
    best_pre = np.nan
    best_rec = np.nan
    best_acc = np.nan

    best_features = env.current_features.copy()
    episode_records = []
    
 
    #Initialize tracking lists
    time_step_list = []
    #reward_calculation_time = []
    time_action_list = []
    #feature_selection_time_list = []
    memory_step_list = []  # New: track memory per step
    episode_time_list = []  # New: track time per episode
    episode_memory_list = []  # New: track memory per episode
    # Memory after initialization
    post_init_memory = get_memory_info()
    print(f"\n===== Memory after initialization =====")
    print(f"RSS: {post_init_memory['rss_gb']:.3f} GB, VMS: {post_init_memory['vms_gb']:.3f} GB")
    print(f"System Memory Usage: {post_init_memory['percent']:.2f}%")
    if torch.cuda.is_available():
        print(f"GPU Memory: {post_init_memory['gpu_gb']:.3f} GB")

    best_performance_records = []
    for episode in range(num_episodes):
        state = env.reset()          
        episode_start_time = time.time()
        episode_start_memory = get_memory_info()
        feature_memory1 = Memory_happo()
        feature_memory2 = Memory_happo()  
        operation_memory = OperationMemory() 
        global_rewards = []
        global_critic_states = []
        global_dones = []
        state = env.reset()          
        episode_step_times = []
        episode_step_memories = []
        

   

        training_start_time = time.time()
        for step in range(max_steps_per_episode):
            step_start_memory = get_memory_info()
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

            feature1 = feature_names[action_f1]     
            feature1_data = feature_data[action_f1] 
            op_state = feature1_data  
            feature1_data_real = env.feature_df[feature1].to_numpy() 
            action_op,logprob_op,entropy_op = operation_agent.select_action(op_state,op_coef=op_coef,action_mask=None) 
            operation = operation_list[action_op]

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

            time_action = end_time_action - start_time_action
            time_action_list.append(time_action)
            info(f'time_action this step: {time_action:.6f}s')

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
            episode_step_times.append(total_step_time)
            #info('#############report 3 function block time consumption per step###############')
            #info(f'Agent action time of this step: {time_action}')
            #info(f'reward calculation time of this step : {reward_cal_step_time}')
            #info(f'Total using time for step-{step+1}: {total_step_time}s')
            
            # Memory tracking per step
            step_end_memory = get_memory_info()
            step_memory_delta = {
                'rss_delta_mb': (step_end_memory['rss_gb'] - step_start_memory['rss_gb']) * 1024,
                'current_rss_gb': step_end_memory['rss_gb'],
                'percent': step_end_memory['percent']
            }
            memory_step_list.append(step_memory_delta)
            episode_step_memories.append(step_memory_delta)
            info(f'Memory this step: RSS={step_end_memory["rss_gb"]:.3f}GB (Δ={step_memory_delta["rss_delta_mb"]:.2f}MB), {step_end_memory["percent"]:.1f}%')




            state = next_state 

            if done:
                break
            

        X_for_mi = env.feature_df[env.current_features]
        y_mi = getattr(env, 'target', target_series)
        #a = 0
        #b = 0
        #a = 10
        #b = 10
        #a = 0.1
        #b = 0.1
        #Rv = a*compute_relevance_Rv(X_for_mi, y_mi)
        #Rd = b*compute_redundancy_Rd(X_for_mi)
        Rv = 0
        Rd = 0
        #info(f'Rv = : {Rv}')
        #info(f'Rd = : {Rd}')                

        #info(f"max(head_actions)={max(int(x) for x in feature_memory1.actions)}")
        #info(f"max(tail_actions)={max(int(x) for x in feature_memory2.actions)}")
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

        with torch.no_grad():
            returns, adv = compute_gae(torch.tensor(global_rewards, dtype=torch.float32),
                                values, gamma=0.99, lam=0.95)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.detach()                     
        
        info(f'✅gae_returns: {returns}')


        #f1-O-f2
        agents_and_mems = [
            ("head", feature_agent_head, feature_memory1),
            ("op",   operation_agent,   operation_memory), 
            ("tail", feature_agent_tail, feature_memory2),
        ]
        #o-f1-f2
        agents_and_mems_2 = [
            ("op",   operation_agent,   operation_memory),
            ("head", feature_agent_head, feature_memory1),
            ("tail", feature_agent_tail, feature_memory2),
        ]
        #O-f2-f1:
        agents_and_mems_3 = [
            ("op",   operation_agent,   operation_memory),
            ("tail", feature_agent_tail, feature_memory2),
            ("head", feature_agent_head, feature_memory1),
        ]
        #f2-O-f1
        agents_and_mems_4 = [
            ("tail", feature_agent_tail, feature_memory2),
            ("op",   operation_agent,   operation_memory), 
            ("head", feature_agent_head, feature_memory1),
        ]
        #f2-f1-O
        agents_and_mems_5 = [
            ("tail", feature_agent_tail, feature_memory2),
            ("head", feature_agent_head, feature_memory1),
            ("op",   operation_agent,   operation_memory),    
        ]
        #f1-f2-O
        agents_and_mems_6 = [
            ("head", feature_agent_head, feature_memory1), 
            ("tail", feature_agent_tail, feature_memory2),
            ("op",   operation_agent,   operation_memory),
        ]


        factor = torch.ones_like(adv)
        ent_coef = 0.01

        '''added'''        
        # Sequential HAPPO updates for each agent
        for agent_name, agent, memory in agents_and_mems:
        #for agent_name, agent, memory in agents_and_mems_2:
        #for agent_name, agent, memory in agents_and_mems_3:
        #for agent_name, agent, memory in agents_and_mems_4:
        #for agent_name, agent, memory in agents_and_mems_5:
        #for agent_name, agent, memory in agents_and_mems_6: 
            info(f"Updating {agent_name} agent with HAPPO...")
            
            if agent_name == "op":
                # Operation agent needs special handling since it has different update method
                # You'll need to add happo_step_iter to OperationAgentPPO class first
                # For now, skip or use its existing update method
                if hasattr(agent, 'happo_step_iter'):
                    loss, factor = agent.happo_step_iter(
                        memory=memory,
                        adv=adv,
                        factor=factor,
                        clip_eps=0.2,
                        ent_coef=ent_coef,
                        max_grad_norm=0.5,
                        Rv=Rv,
                        Rd=Rd
                    )
                    info(f"{agent_name} agent loss: {loss:.4f}")
                else:
                    info(f"Skipping {agent_name} agent - no HAPPO method implemented")
            else:
                # Feature agents (head and tail) use happo_step_iter
                loss, factor = agent.happo_step_iter(
                    memory=memory,
                    adv=adv,
                    factor=factor,
                    clip_eps=0.2,
                    ent_coef=ent_coef,
                    max_grad_norm=0.5,
                    Rv=Rv,
                    Rd=Rd
                )
                info(f"{agent_name} agent loss: {loss:.4f}")
        

        #critic update
        critic_optimizer.zero_grad()
        pred_v = values[:-1]  # [T]
        value_loss = 0.5 * torch.mean((pred_v - returns)**2)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(shared_critic.parameters(), 0.5)
        critic_optimizer.step()
        


        
        if task_type == 'reg':
            mse, rae = evaluate_features(env, env.current_features, task_type='reg',n_splits=n_splits)
            print(f'Episode {episode+1}, MSE: {mse}, 1-RAE:{rae}')
        else:
            f1, cross_entropy = evaluate_features(env, env.current_features, task_type='cls',n_splits=n_splits)
            print(f'Episode {episode+1}, F1: {f1}, Cross Entropy:{cross_entropy}')


        #top_k = 20       
        #best_top_K_features, metrics = select_topk_features_and_evaluate(env.feature_df, y, top_k, task_type, n_splits=n_splits)
        #top_k = 150
        #higgs 0.709, NIPS137
        #top_k = 150
        top_k = params['top_k']
        best_top_K_features, metrics = select_topk_features_and_evaluate(env.feature_df_test, y_test, top_k, task_type, n_splits=n_splits)
        top_k_used = top_k

        #all_features_for_eval = list(dict.fromkeys(          
        #    env.original_features_test + best_top_K_features
        #))
        #metrics = evaluate_features_new(
        #    env, all_features_for_eval, task_type=task_type, n_splits=n_splits,y_test=y_test
        #)

        if task_type == 'reg':
            current_RAE = metrics['mean_RAE']
            if current_RAE > best_RAE:
                best_features_global = best_top_K_features.copy()
                best_MAE = metrics['mean_MAE']
                best_MSE = metrics['mean_MSE']
                best_R2  = metrics['mean_R2']
                best_RAE = current_RAE
            
                elapsed_time = time.time() - total_start_time
                best_performance_records.append({
                    'episode': episode + 1,
                    'step_in_episode': step + 1,
                    'elapsed_time_sec': elapsed_time,
                    'elapsed_time_min': elapsed_time / 60,
                    'best_1MAE': best_MAE,
                    'best_1MSE': best_MSE,
                    'best_1RAE': best_RAE,
                    'best_R2': best_R2,
                    'num_features_selected': len(best_features_global),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                info(f'🎯 New best performance at episode {episode+1}! Elapsed time: {elapsed_time:.2f} s')


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

                elapsed_time = time.time() - total_start_time
                best_performance_records.append({
                    'episode': episode + 1,
                    'step_in_episode': step + 1,
                    'elapsed_time_sec': elapsed_time,
                    'elapsed_time_min': elapsed_time / 60,
                    'best_precision': best_pre,
                    'best_recall': best_rec,
                    'best_f1': best_F1,
                    'best_accuracy': best_acc,
                    'num_features_selected': len(best_features_global),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                info(f'🎯 New best performance at episode {episode+1}! Elapsed time: {elapsed_time:.2f} s')
            
            
            episode_records.append({
            'episode': episode + 1,
            'current_reward': current_F1,
            'best_reward': best_F1
            })


        # Episode-level tracking
        episode_end_time = time.time()
        episode_end_memory = get_memory_info()
        
        episode_time = episode_end_time - episode_start_time
        episode_memory_delta = (episode_end_memory['rss_gb'] - episode_start_memory['rss_gb']) * 1024  # MB
        
        episode_time_list.append(episode_time)
        episode_memory_list.append({
            'start_rss_gb': episode_start_memory['rss_gb'],
            'end_rss_gb': episode_end_memory['rss_gb'],
            'delta_mb': episode_memory_delta,
            'end_percent': episode_end_memory['percent'],
            'num_steps': len(episode_step_times),
            'avg_step_time': np.mean(episode_step_times) if episode_step_times else 0,
            'avg_step_memory_mb': np.mean([m['rss_delta_mb'] for m in episode_step_memories]) if episode_step_memories else 0
        })


        if task_type == 'reg':
            info(f"episode {episode + 1} ; top_k features: {best_top_K_features} ; current best Metrics: Best 1-MAE {best_MAE}; Best 1 - MSE {best_MSE}; Best 1-RAE {best_RAE}")
        elif task_type == 'cls':
            info(f"episode {episode + 1} ; top_k features: {best_top_K_features} ; current best Metrics: Best Pre {best_pre}; Best rec {best_rec}; Best F1 {best_F1}")
        info(f'Total using time for episode {episode+1}: {time.time() - training_start_time:.1f}s')


        print(f"\n===== Episode {episode+1} Summary =====")
        print(f"Time: {episode_time:.2f}s, Steps: {len(episode_step_times)}, Avg step: {np.mean(episode_step_times):.4f}s")
        print(f"Memory: Start={episode_start_memory['rss_gb']:.3f}GB, End={episode_end_memory['rss_gb']:.3f}GB, Delta={episode_memory_delta:.2f}MB")
        print(f"System Memory Usage: {episode_end_memory['percent']:.1f}%")
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
    
    # Get only the best features to save to CSV
    #best_feature_df = env.feature_df[best_features_global].copy()
    best_feature_df = env.feature_df_test[best_features_global].copy()
    #best_feature_df["label"] = env.target
    best_feature_df["label"] = y_test

    result_dir = params['result_dir']
    dataset_name = params['dataset']
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename_csv = os.path.join(result_dir, f"{dataset_name}_{timestamp}_best_features_{metric_suffix}.csv")
    best_feature_df.to_csv(filename_csv, index=False)
    dataset_dir = os.path.dirname(filename_csv)
    records_filename = os.path.join(dataset_dir, f"{dataset_name}_{timestamp}_episode_rewards.csv")
    records_df = pd.DataFrame(episode_records)
    records_df.to_csv(records_filename, index=False)
    print(f"Best feature DataFrame (with label) saved to {filename_csv}")
    print(f"Episode rewards DataFrame saved to {records_filename}")

    logger.info(f"Dataset {dataset_name} completed successfully. Results saved to {filename_csv}")
    print(f"Dataset {dataset_name} completed successfully. Results saved to {filename_csv}")
    
    # End time of all episodes 
    total_end_time = time.time()  
    total_time_spent = total_end_time - total_start_time  
    info(f'Total time spent for all episodes: {total_time_spent:.1f}s')  

    # Final memory state
    final_memory = get_memory_info()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    info(f'Total time spent for all episodes: {total_time_spent:.1f}s')

    # Compute and save timing statistics
    avg_step_time = float(np.mean(time_step_list)) if len(time_step_list) > 0 else 0.0
    avg_action_time = float(np.mean(time_action_list)) if len(time_action_list) > 0 else 0.0
    steps_recorded = int(len(time_step_list))

    avg_episode_time = float(np.mean(episode_time_list)) if episode_time_list else 0.0
    total_memory_growth = (final_memory['rss_gb'] - initial_memory['rss_gb']) * 1024  # MB

    # Memory statistics
    avg_memory_per_episode = float(np.mean([e['delta_mb'] for e in episode_memory_list])) if episode_memory_list else 0.0
    max_memory_usage = max([e['end_rss_gb'] for e in episode_memory_list]) if episode_memory_list else 0.0

    # Final selected feature count (safe fallback if not set by any improvement)
    try:
        final_selected_feature_count = len(best_features_global)
    except Exception:
        final_selected_feature_count = np.nan

    if best_performance_records:
        best_perf_df = pd.DataFrame(best_performance_records)
        best_perf_filename = os.path.join(result_dir, f"{dataset_name}_{timestamp}_best_performances.csv")
        best_perf_df.to_csv(best_perf_filename, index=False)
        print(f"Best performance records saved to {best_perf_filename}")
        
        info("\n" + "="*60)
        info("BEST PERFORMANCE TRACKING SUMMARY")
        info("="*60)
        info(f"Total improvements: {len(best_performance_records)}")
        
        for i, record in enumerate(best_performance_records, 1):
            info(f"\nImprovement #{i}:")
            info(f"  Episode: {record['episode']}, Step: {record['step_in_episode']}")
            info(f"  Time: {record['elapsed_time_min']:.2f} min ({record['elapsed_time_sec']:.1f} sec)")
            info(f"  Timestamp: {record['timestamp']}")
            if task_type == 'reg':
                info(f"  Metrics: 1-MAE={record['best_1MAE']:.4f}, 1-MSE={record['best_1MSE']:.4f}, 1-RAE={record['best_1RAE']:.4f}")
            else:
                info(f"  Metrics: P={record['best_precision']:.4f}, R={record['best_recall']:.4f}, F1={record['best_f1']:.4f}")
            info(f"  Features selected: {record['num_features_selected']}")
        
        info("\n" + "="*60)
    else:
        info("No performance improvements were recorded during training.")


    stats = {
        'dataset': dataset_name,
        'ap_k': int(params.get('ap_k', 100)) if dataset_name == 'ap_omentum_ovary' else None,
        'num_episodes': num_episodes,
        'max_steps': max_steps_per_episode,
        'steps_recorded': steps_recorded,
        'avg_step_time_sec': avg_step_time,
        'avg_action_time_sec': avg_action_time,
        'avg_episode_time_sec': avg_episode_time,
        'total_time_sec': float(total_time_spent),
        # Memory statistics
        'initial_memory_gb': initial_memory['rss_gb'],
        'final_memory_gb': final_memory['rss_gb'],
        'total_memory_growth_mb': total_memory_growth,
        'avg_memory_per_episode_mb': avg_memory_per_episode,
        'max_memory_usage_gb': max_memory_usage,
        'peak_traced_memory_mb': peak / 1024**2,  # Convert bytes to MB
        # Feature statistics
        'top_k_used': top_k_used if 'top_k_used' in locals() else params.get('top_k', None),
        'final_selected_feature_count': final_selected_feature_count,
        'timestamp': timestamp,

        # Best performance tracking with timing
        'num_improvements': len(best_performance_records),
        'first_improvement_time_min': best_performance_records[0]['elapsed_time_min'] if best_performance_records else None,
        'final_improvement_time_min': best_performance_records[-1]['elapsed_time_min'] if best_performance_records else None,

    }
    # Attach performance metrics alongside timing
    if task_type == 'reg':
        stats.update({
            'best_1MAE': best_MAE,
            'best_1MSE': best_MSE,
            'best_1RAE': best_RAE,
            'best_R2':   best_R2,
            'original_1MAE': original_MAE,
            'original_1MSE': original_MSE,
            'original_1RAE': original_RAE,
        })
    elif task_type == 'cls':
        stats.update({
            'best_precision': best_pre,
            'best_recall':    best_rec,
            'best_f1':        best_F1,
            'best_accuracy':  best_acc,
            'original_precision': original_pre,
            'original_recall':    original_rec,
            'original_f1':        original_f1,
        })
    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(result_dir, f"{dataset_name}_{timestamp}_time_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    # Save detailed episode tracking
    episode_tracking = []
    for i, episode_mem in enumerate(episode_memory_list):
        episode_tracking.append({
            'episode': i + 1,
            'episode_time_sec': episode_time_list[i] if i < len(episode_time_list) else None,
            'start_memory_gb': episode_mem['start_rss_gb'],
            'end_memory_gb': episode_mem['end_rss_gb'],
            'memory_delta_mb': episode_mem['delta_mb'],
            'system_memory_percent': episode_mem['end_percent'],
            'num_steps': episode_mem['num_steps'],
            'avg_step_time_sec': episode_mem['avg_step_time'],
            'avg_step_memory_mb': episode_mem['avg_step_memory_mb']
        })

    if episode_tracking:
        episode_df = pd.DataFrame(episode_tracking)
        episode_path = os.path.join(result_dir, f"{dataset_name}_{timestamp}_episode_tracking.csv")
        episode_df.to_csv(episode_path, index=False)
        print(f"Episode tracking saved to {episode_path}")
    
    # Print comprehensive summary
    # Print comprehensive summary using info
    info("\n" + "="*60)
    info("EXPERIMENT COMPLETE - FINAL REPORT")
    info("="*60)
    
    info("\n===== Timing Summary =====")
    info(f"Dataset: {stats['dataset']}  ap_k: {stats['ap_k']}")
    info(f"Episodes: {stats['num_episodes']}  MaxSteps/Ep: {stats['max_steps']}  StepsRecorded: {stats['steps_recorded']}")
    info(f"Total time: {stats['total_time_sec']:.2f}s")
    info(f"Avg episode time: {stats['avg_episode_time_sec']:.2f}s")
    info(f"Avg step time: {stats['avg_step_time_sec']:.6f}s")
    info(f"Avg action time: {stats['avg_action_time_sec']:.6f}s")
    
    info("\n===== Memory Summary =====")
    info(f"Initial memory: {stats['initial_memory_gb']:.3f} GB")
    info(f"Final memory: {stats['final_memory_gb']:.3f} GB")
    info(f"Total growth: {stats['total_memory_growth_mb']:.2f} MB")
    info(f"Avg per episode: {stats['avg_memory_per_episode_mb']:.2f} MB")
    info(f"Peak memory: {stats['max_memory_usage_gb']:.3f} GB")
    info(f"Peak traced: {stats['peak_traced_memory_mb']:.2f} MB")
    
    info("\n===== Feature Selection =====")
    info(f"top_k_used: {stats['top_k_used']}")
    info(f"Final selected features: {stats['final_selected_feature_count']}")
    
    info("\n===== Performance Metrics =====")
    if task_type == 'reg':
        info("Best Performance:")
        info(f"  1-MAE: {stats.get('best_1MAE', 'N/A')}")
        info(f"  1-MSE: {stats.get('best_1MSE', 'N/A')}")
        info(f"  1-RAE: {stats.get('best_1RAE', 'N/A')}")
        info(f"  R2: {stats.get('best_R2', 'N/A')}")
        info("Original Performance:")
        info(f"  1-MAE: {stats.get('original_1MAE', 'N/A')}")
        info(f"  1-MSE: {stats.get('original_1MSE', 'N/A')}")
        info(f"  1-RAE: {stats.get('original_1RAE', 'N/A')}")
    elif task_type == 'cls':
        info("Best Performance:")
        info(f"  Precision: {stats.get('best_precision', 'N/A')}")
        info(f"  Recall: {stats.get('best_recall', 'N/A')}")
        info(f"  F1: {stats.get('best_f1', 'N/A')}")
        info(f"  Accuracy: {stats.get('best_accuracy', 'N/A')}")
        info("Original Performance:")
        info(f"  Precision: {stats.get('original_precision', 'N/A')}")
        info(f"  Recall: {stats.get('original_recall', 'N/A')}")
        info(f"  F1: {stats.get('original_f1', 'N/A')}")
    
    print(f"\nAll statistics saved to {stats_path}")
    print("="*60)

    """
    # Print a concise summary to terminal
    print("\n===== Timing Summary =====")
    print(f"Dataset: {stats['dataset']}  ap_k: {stats['ap_k']}")
    print(f"Episodes: {stats['num_episodes']}  MaxSteps/Ep: {stats['max_steps']}  StepsRecorded: {stats['steps_recorded']}")
    print(f"Avg step time: {stats['avg_step_time_sec']:.6f}s  Avg action time: {stats['avg_action_time_sec']:.6f}s  Total: {stats['total_time_sec']:.2f}s")
    print(f"top_k_used: {stats['top_k_used']}  final_selected_feature_count: {stats['final_selected_feature_count']}")
    if task_type == 'reg':
        print(f"Best (1-MAE/1-MSE/1-RAE/R2): {stats.get('best_1MAE')}, {stats.get('best_1MSE')}, {stats.get('best_1RAE')}, {stats.get('best_R2')}")
        print(f"Original (1-MAE/1-MSE/1-RAE): {stats.get('original_1MAE')}, {stats.get('original_1MSE')}, {stats.get('original_1RAE')}")
    elif task_type == 'cls':
        print(f"Best (P/R/F1/Acc): {stats.get('best_precision')}, {stats.get('best_recall')}, {stats.get('best_f1')}, {stats.get('best_accuracy')}")
        print(f"Original (P/R/F1): {stats.get('original_precision')}, {stats.get('original_recall')}, {stats.get('original_f1')}")
    print(f"Timing stats saved to {stats_path}")
    """
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
