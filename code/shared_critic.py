# shared_critic.py

import torch
import torch.nn as nn
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

seed_value = 0
random.seed(seed_value)
#np.random.seed(seed_value)
torch.manual_seed(seed_value)


'''
class SharedCritic(nn.Module):
    def __init__(self, input_dim):
        super(SharedCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_value = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        state_value = self.fc_value(x)
        return state_value
'''

class SharedCritic(nn.Module):
    def __init__(self, input_dim):
        super(SharedCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc_value = nn.Linear(64, 1)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        state_value = self.fc_value(x)
        return state_value





class FeatureSetAttention(nn.Module):

    def __init__(self, feature_dim=7, embed_dim=64, num_heads=4, hidden_dim=128):
        super(FeatureSetAttention, self).__init__()
        
        # Token embedding layer - maps each feature to continuous space
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(), 
            #nn.LeakyReLU(negative_slope=0.01)
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.attention_pooling = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )
        

        self.output_projection = nn.Linear(embed_dim, hidden_dim)
        
    def forward(self, feature_data):

        if feature_data.dim() == 2:
            feature_data = feature_data.unsqueeze(0)  # [1, num_features, feature_dim]
        
        batch_size, num_features, _ = feature_data.shape
        
        embeddings = self.feature_embedding(feature_data)  # [batch, num_features, embed_dim]
        
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        attended = self.layer_norm(embeddings + attended)  # Residual connection
        
        attention_weights = self.attention_pooling(attended)  # [batch, num_features, 1]
        pooled = torch.sum(attention_weights * attended, dim=1)  # [batch, embed_dim]
        
        interaction_vector = self.output_projection(pooled)  # [batch, hidden_dim]
        print("Interaction vector:", interaction_vector)
        return interaction_vector.squeeze(0) if batch_size == 1 else interaction_vector