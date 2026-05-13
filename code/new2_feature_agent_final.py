# feature_agent.py

import random
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

#Custome import
from utils.logger import *
from utils.tools import calculate_meta_statistics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

logger = logging.getLogger(__name__)
logger.propagate = True

O1 = ['sqrt', 'square', 'sin', 'cos', 'tanh', 
        'cube', 'exp', 'sigmoid', 'log', 'reciprocal']
O2 = ['+', '-', '*', '/']
O3 = ['stand_scaler', 'minmax_scaler', 'quan_trans']
operation_list = O1 + O2 + O3
one_hot_op = pd.get_dummies(operation_list)
operation_emb = defaultdict()
for item in one_hot_op.columns:
    operation_emb[item] = torch.tensor(one_hot_op[item].values, dtype=torch.float32)
    
class FeatureAgentPPO(nn.Module):
    def __init__(self, feature_dim,select='head',operation_dim = None):
        super(FeatureAgentPPO, self).__init__()
        self.feature_dim = feature_dim
        self.select = select
        self.operation_dim = operation_dim 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=7, batch_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2) 
        self.fc_policy = nn.Linear(feature_dim, 1)

    def forward(self, x):
        x_encoded = self.transformer_encoder(x)
        if self.select == 'tail':
            print("x_encoded")
            print("--"*20)
            x_selectable = x_encoded[:, :-1, :]  
            info(f"x_encoded.shape={x_encoded.shape}, x_selectable.shape={(x_encoded[:, :-1, :]).shape}")
            action_logits = self.fc_policy(x_selectable).squeeze(-1)
        else:
            action_logits = self.fc_policy(x_encoded).squeeze(-1)
        action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs

class PPOAgent:
    def __init__(self, feature_dim, shared_critic,
                 feature_lr, gamma=0.99, eps_clip=0.2,select = 'head',operation_dim = None,operation_emb = None):
        self.policy = FeatureAgentPPO(feature_dim)
        self.shared_critic = shared_critic   
        self.select = select
        self.feature_dim = feature_dim
        self.operation_dim = operation_dim
        self.operation_emb = operation_emb

        if self.select == 'tail':
            assert operation_dim is not None, "operation_dim must be specified when select='tail'"
            self.operation_dim = operation_dim
            self.hidden_dim = 256  
            self.projection_layer = nn.Sequential(
                nn.Linear(self.feature_dim + self.operation_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim)
            )
        else:
            self.projection_layer = None  
        
        if self.select == 'tail':
            self.optimizer = optim.Adam(
                list(self.policy.parameters()) + list(self.projection_layer.parameters()), lr=feature_lr)
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=feature_lr)
                 
        self.gamma = gamma
        self.eps_clip = eps_clip 
    def _select_head(self, feature_data):
        action_probs = self.policy(feature_data)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze(0) 

    def _select_tail(self, feature_data,state_emb,op):

        op_emb = self.get_op_emb(op) 
        feature_data_tensor = torch.tensor(state_emb, dtype=torch.float32)
        feature_data_tensor = feature_data_tensor.unsqueeze(0) 
        op_emb = op_emb.unsqueeze(0)  
        state_op_emb = torch.cat((feature_data_tensor, op_emb),dim = 1) 
        feature_op_emb = self.feature_op_map(state_op_emb) 
        #feature_data_op = feature_data + feature_op_emb.unsqueeze(0)  
        action_probs = self.policy(feature_data) 
        #action_probs = self.policy(feature_data_op) 
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.squeeze(0)

    def get_op_emb(self, op):
        if type(op) is int:
            assert op >= 0 and op < len(operation_list)
            return self.operation_emb[operation_list[op]]
        elif type(op) is str:
            assert op in operation_list
            return self.operation_emb[op]
        else:  
            return op

    def feature_op_map(self, state_op_emb):

        return self.projection_layer(state_op_emb)
            
    def update(self, memory, critic_optimizer, critic_input_dim):
        rewards = torch.tensor(memory.rewards, dtype=torch.float32) 
        old_states = [torch.tensor(s, dtype=torch.float32)
                      for s in memory.states]
        old_actions = torch.tensor(memory.actions)
        old_logprobs = torch.stack(memory.logprobs)   
        
        discounted_rewards = []
        Gt = 0
        for reward in reversed(rewards.tolist()): 
            Gt = reward + self.gamma * Gt
            discounted_rewards.insert(0, Gt)
        discounted_rewards = torch.tensor(discounted_rewards,
                                          dtype=torch.float32)
        '''
        discounted_rewards = (discounted_rewards -
                              discounted_rewards.mean()) / \
                             (discounted_rewards.std() + 1e-5)
        '''
        total_policy_loss = 0
        total_value_loss = 0
        info(f"discounted_rewards: {discounted_rewards}")
        for i in range(len(rewards)):
            feature_data = old_states[i]                               
            action = old_actions[i]
            old_logprob = old_logprobs[i]
            reward = discounted_rewards[i]       
            
            feature_data = feature_data.unsqueeze(0)
            
            action_probs = self.policy(feature_data)
            dist = Categorical(action_probs)
            logprob = dist.log_prob(action)
            entropy = dist.entropy()

            feature_data_squeezed = feature_data.squeeze(0)
            state = calculate_meta_statistics(feature_data_squeezed.numpy())
            state = state.flatten()   
            if torch.any(torch.tensor(state) > 10):
                info(f"High value detected in flattened state: {state}, proceeding with caution!!!")

            state_tensor = torch.tensor(state, dtype=torch.float32)
            state_tensor = state_tensor.unsqueeze(0)
            state_value = self.shared_critic(state_tensor)
            if state_value > 10:
                info(f"High state_value detected: {reward}, proceeding with caution.!!!")
            if reward > 10:
                info(f"High reward detected: {reward}, proceeding with caution.!!!")
           
            advantage = reward - state_value.item()                    
            ratio = torch.exp(logprob - old_logprob.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio,
                                1 - self.eps_clip,
                                1 + self.eps_clip) * advantage    
            policy_loss = -torch.min(surr1, surr2) - 0.1 * entropy.mean()
            total_policy_loss += policy_loss
            info(f'PPO Agent total_policy_loss: {total_policy_loss}')
            value_loss = 0.5 * (state_value - reward) ** 2
            info(f'central critic value_loss for ft: {value_loss}')
            total_value_loss += value_loss

        self.optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        critic_optimizer.zero_grad()
        total_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.shared_critic.parameters(), max_norm=0.5) 
        critic_optimizer.step()

        return total_policy_loss.item(), total_value_loss.item()




    def happo_step_iter(self, memory, adv, factor,
                        clip_eps=None, ent_coef=0.1, max_grad_norm=0.5):
        if self.select == 'tail':
            info("currently update the tail agent, for debug reason")
        else:
            info("currently update the head agent, for debug reason")
        
        #info(f"check the initial_factor returned {factor}")

        info("==== [DEBUG] happo_step_iter input snapshot for feature Agent ====")
        info(f"clip_eps={clip_eps}, ent_coef={ent_coef}, max_grad_norm={max_grad_norm}")
        info(f"adv:    shape={tuple(adv.shape)}, dtype={adv.dtype}, requires_grad={adv.requires_grad}")
        info(f"factor: shape={tuple(factor.shape)}, dtype={factor.dtype}, requires_grad={factor.requires_grad}")
        info(f"len(obs)={len(memory.obs)}, len(actions)={len(memory.actions)}, "
            f"len(logprobs)={len(memory.logprobs)}, len(mask)={len(memory.mask)}")


        eps_clip = self.eps_clip
        if clip_eps is None: clip_eps = self.eps_clip
        device = next(self.policy.parameters()).device

        # Information retrieval from memory - similar to update method
        old_states = [torch.tensor(s, dtype=torch.float32) for s in memory.obs]
        old_actions = torch.tensor(memory.actions)
        old_logprobs = torch.stack(memory.logprobs)
        
        mask = memory.mask  # assuming mask is stored in memory

        total_loss = torch.zeros(1, device=device)

        ratios_det = []

        self.optimizer.zero_grad()

        info(f"Processing {len(old_states)} states from memory")

        for i in range(len(old_states)):
            # Extract data for this timestep - similar to update method loop
            feature_data = old_states[i]
            action = old_actions[i]
            old_logprob = old_logprobs[i]
            
            #info(f"Step {i} - feature_data shape: {feature_data.shape}")
            #info(f"Step {i} - action: {action}")

            # Ensure proper batch dimension
            feature_data = feature_data.unsqueeze(0)

            # Policy forward pass
            action_probs = self.policy(feature_data)
            dist = Categorical(action_probs)
            logprob = dist.log_prob(action)
            entropy = dist.entropy()

            # Get advantage and factor for this timestep
            adv_t = adv[i].to(device)
            f_t = factor[i].to(device)
            
            info(f"Step {i} - adv_t: {adv_t}, f_t: {f_t}")

            # PPO loss calculation
            ratio = torch.exp(logprob - old_logprob.detach())
            s1 = f_t * ratio * adv_t
            s2 = f_t * torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_t
            loss_t = -(torch.min(s1, s2)) - ent_coef * entropy
            total_loss += loss_t

            #ratios_det.append(ratio.detach())
            ratios_det.append(ratio.detach().view(()))  # 明确变成标量 0-d tensor

            info(f"Step {i} - ratio: {ratio.item()}, loss_t: {loss_t.item()}")

        # Average over valid steps
        valid_steps = max(sum(mask), 1.0) if hasattr(memory, 'mask') else len(old_states)
        total_loss = total_loss / valid_steps

        info(f"Total loss before backward: {total_loss.item()}")

        # Backward pass and optimization
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.optimizer.step()

        # Update factor (element-wise)
        with torch.no_grad():
            #info(f"check the ratios {ratio}")
            #info(f"check the ratios_det {ratios_det}")
            #ratios_det = torch.stack(ratios_det, dim=0).to(factor.device)
            ratios_det = torch.stack(ratios_det, dim=0).to(factor.device).view(-1)
            if hasattr(memory, 'mask'):
                #mask_t = torch.tensor(mask, dtype=torch.float32, device=factor.device)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=factor.device).view(-1)
                new_factor = factor * ratios_det * mask_t + factor * (1 - mask_t)
            else:
                new_factor = factor * ratios_det


        return total_loss.item(), new_factor