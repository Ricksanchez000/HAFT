# operation_agent.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from utils.logger import *
from utils.tools import calculate_meta_statistics

seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

logger = logging.getLogger(__name__)
logger.propagate = True

class OperationAgentA2C(nn.Module):
    def __init__(self, input_dim, action_dim): 
        super(OperationAgentA2C, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(128, 64)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.fc_policy = nn.Linear(64, action_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc_policy.weight)
        nn.init.zeros_(self.fc_policy.bias)

    def forward(self, x):
        x = self.leaky_relu1(self.fc1(x))
        x = self.leaky_relu2(self.fc2(x))
        logits = self.fc_policy(x) 
        logits = logits - logits.max(dim=-1, keepdim=True)[0] 
        return logits
    

class OperationAgentPPO:
    def __init__(self, input_dim, action_dim, shared_critic,
                 lr=1e-4, gamma=0.99):
        self.policy = OperationAgentA2C(input_dim, action_dim)
        self.shared_critic = shared_critic
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.gamma = gamma
        self.action_dim = action_dim
        self.eps_clip = 0.2

    def select_action(self, state, op_coef,action_mask=None):

        if action_mask is not None:
            if not isinstance(action_mask, np.ndarray):
                action_mask = np.array(action_mask)
        else:
            action_mask = np.ones(self.action_dim, dtype=int) 
  
        print("State:", state)
        state = np.concatenate([state, action_mask]) 
        print("Action Mask:", action_mask)
        if np.isnan(state).any() or np.isinf(state).any():
            info(f'State contains NaN or Inf values.')   
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.policy(state_tensor)  
        print("Logits:", logits)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            info(f'Logits contain NaN or Inf values.')
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)  
        logits[~action_mask_tensor] = -1e15
        if (action_mask_tensor.sum() == 0).item():
            raise ValueError("All actions are masked out! Please check your code.")
        action_probs = torch.softmax(logits, dim=-1)
        m = Categorical(action_probs)
        entropy = m.entropy().mean()  
        entropy_coef = op_coef   
        entropy_term = entropy_coef * entropy
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob.squeeze(0), entropy_term

    def update(self, rewards, log_probs, states,
               critic_optimizer, critic_input_dim,entropy_op,actions,operation_action_mask,operation_action_state):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_actions = torch.tensor(actions)
        old_logprobs = torch.stack(log_probs)
        #operation_action_mask = torch.tensor(operation_action_mask, dtype=torch.bool)
        #operation_action_state = torch.tensor(operation_action_state, dtype=torch.float32)
        operation_action_mask = operation_action_mask
        operation_action_state = operation_action_state
        self.eps_clip=0.2

        discounted_rewards = []
        Gt = 0
        for reward in reversed(rewards.tolist()):
            Gt = reward + self.gamma * Gt
            discounted_rewards.insert(0, Gt)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        total_policy_loss = 0
        total_value_loss = 0


        
        for i in range(len(rewards)):
            
            state = states[i]
            action = old_actions[i]
            old_log_prob = old_logprobs[i]  
            reward = discounted_rewards[i]


            
            action_mask = operation_action_mask[i] 
            action_state = operation_action_state[i]
            action_state = np.concatenate([action_state, action_mask])
            state_tensor = torch.tensor(action_state, dtype=torch.float32).unsqueeze(0)
            logits = self.policy(state_tensor)
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)  
            logits[~action_mask_tensor] = -1e15
            action_probs = torch.softmax(logits, dim=-1) 
            
            
            processed_state = calculate_meta_statistics(state) 
            processed_state = processed_state.flatten()
            state_tensor = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)


            
            #action_probs = self.policy(state_tensor)
            dist = Categorical(action_probs)
            new_log_prob = dist.log_prob(action)
            entropy = dist.entropy()  
            
            state_value = self.shared_critic(state_tensor)

            
            advantage = discounted_rewards[i] - state_value.detach().item()
            
            
            ratio = torch.exp(new_log_prob - old_log_prob.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2) - 0.1 * entropy.mean()
            total_policy_loss += policy_loss

             
            value_loss = 0.5 * (state_value - discounted_rewards[i]) ** 2
            total_value_loss += value_loss
        
        
        self.optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        
        critic_optimizer.zero_grad()
        total_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.shared_critic.parameters(), max_norm=0.5)
        critic_optimizer.step()
        
        #return total_policy_loss.item(), total_value_loss.item()

    def happo_step_iter(self, memory, adv, factor,
                        clip_eps=None, ent_coef=0.1, max_grad_norm=0.5,Rv=0,Rd=0):


        info("update the Operation Agent")

        if clip_eps is None: clip_eps = self.eps_clip
        device = next(self.policy.parameters()).device
        

        # ---------- DEBUG BLOCK BEGIN ----------
        info("==== [DEBUG] happo_step_iter input snapshot for Operation Agent ====")
        info(f"clip_eps={clip_eps}, ent_coef={ent_coef}, max_grad_norm={max_grad_norm}")
        info(f"adv:    shape={tuple(adv.shape)}, dtype={adv.dtype}, requires_grad={adv.requires_grad}")
        info(f"factor: shape={tuple(factor.shape)}, dtype={factor.dtype}, requires_grad={factor.requires_grad}")
        info(f"len(obs)={len(memory.obs)}, len(actions)={len(memory.actions)}, "
            f"len(logprobs)={len(memory.logprobs)}, len(mask)={len(memory.mask)}")

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
            #info(f"Step {i} - ratio: {ratio.item()}, loss_t: {loss_t.item()}")

        # Average over valid steps
        valid_steps = max(sum(mask), 1.0) if hasattr(memory, 'mask') else len(old_states)
        total_loss = total_loss / valid_steps + Rd - Rv
        #total_loss = total_loss / valid_steps

        info(f"Total loss before backward: {total_loss.item()}")

        # Backward pass and optimization
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.optimizer.step()

        # Update factor (element-wise)
        with torch.no_grad():
            #ratios_det = torch.stack(ratios_det, dim=0).to(factor.device)
            ratios_det = torch.stack(ratios_det, dim=0).to(factor.device).view(-1)
            if hasattr(memory, 'mask'):
                #mask_t = torch.tensor(mask, dtype=torch.float32, device=factor.device)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=factor.device).view(-1)#
                new_factor = factor * ratios_det * mask_t + factor * (1 - mask_t)
            else:
                new_factor = factor * ratios_det

        #info(f"😒check the new_factor returned")
        return total_loss.item(), new_factor
