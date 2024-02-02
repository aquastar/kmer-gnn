import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, SAGEConv, GATv2Conv, APPNP
from torch_geometric.nn import global_add_pool

hidden_dim = 16


def global_attention_pool(x, batch, alpha=0.2):
    """
    Global Attention Pooling function.

    Args:
        x (Tensor): Node feature matrix.
        batch (Tensor): Batch assignment tensor.
        alpha (float): Alpha value for attention scores.

    Returns:
        Tensor: Pooled node features.
    """
    num_nodes = x.size(0)

    # Calculate attention scores
    attention_scores = torch.exp(alpha * x)
    attention_scores = attention_scores / scatter_add(attention_scores, batch, dim=0)[batch]

    # Apply attention to node features
    x = x * attention_scores

    # Sum the attended node features for each graph in the batch
    pooled_x = scatter_add(x, batch, dim=0)

    return pooled_x

class GCNRegression(torch.nn.Module):
    def __init__(self, num_layers, hidden_dim=hidden_dim):
        super(GCNRegression, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(1, hidden_dim))
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Add attention-based pooling layer
        self.pool = global_attention_pool

        # Add linear layers after each convolutional layer
        self.linears = nn.ModuleList()
        for i in range(num_layers):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))

        # Replace self.linear1 with three linear layers
        self.linear1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

            # Add linear layer after convolutional layer
            x = self.linears[i](x)

        # Apply attention-based pooling
        x = self.pool(x, batch)

        # Apply the final linear layers
        x = self.linear1(x)

        return x
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=hidden_dim, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, batch = data.x, data.batch
        x = x.view(torch.max(batch).item()+1, -1)  # Flatten the input tensor

        # Apply the first linear layer, activation function, and dropout
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        
        # Apply the second linear layer, activation function, and dropout
        x = F.relu(self.linear2(x))
        x = self.dropout(x)

        # Apply the third linear layer and dropout
        x = F.relu(self.linear3(x))
        x = self.dropout(x)

        # Apply the output layer
        x = self.output_layer(x)

        return x

class GATRegression(torch.nn.Module):
    def __init__(self, num_layers, hidden_dim=hidden_dim):
        super(GATRegression, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(1, hidden_dim))
        for i in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim))
        
        # Add linear layers after each convolutional layer
        self.linears = nn.ModuleList()
        for i in range(num_layers):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))

        # Replace self.linear1 with three linear layers
        self.linear1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

            # Add linear layer after convolutional layer
            x = self.linears[i](x)

        # Apply attention-based pooling (global_attention_pool)
        x = global_attention_pool(x, batch)

        # Apply the final linear layers
        x = self.linear1(x)

        return x



class SAGERegression(torch.nn.Module):
    def __init__(self, num_layers, hidden_dim=hidden_dim):
        super(SAGERegression, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(1, hidden_dim, aggr='mean'))
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
        
        # Add linear layers after each convolutional layer
        self.linears = nn.ModuleList()
        for i in range(num_layers):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))

        # Replace self.out with three linear layers
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            
            # Add linear layer after convolutional layer
            x = self.linears[i](x)

        # Apply attention-based pooling (global_attention_pool)
        x = global_attention_pool(x, data.batch)

        # Apply the final linear layers
        x = self.out(x)

        return x

class GATv2Regression(torch.nn.Module):
    def __init__(self, num_layers, hidden_dim=hidden_dim):
        super(GATv2Regression, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(1, hidden_dim))
        for i in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim))
        
        # Add linear layers after each convolutional layer
        self.linears = nn.ModuleList()
        for i in range(num_layers):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))

        # Replace self.out with three linear layers
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            
            # Add linear layer after convolutional layer
            x = self.linears[i](x)

        # Apply attention-based pooling (global_add_pool)
        x = global_add_pool(x, data.batch)

        # Apply the final linear layers
        x = self.out(x)

        return x


# Define APPNP Model
class APPNPRegression(torch.nn.Module):
    def __init__(self, num_layers, K=10, alpha=0.1):
        super(APPNPRegression, self).__init__()
        self.lin1 = nn.Linear(1, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, 1)
        self.prop1 = APPNP(K, alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.prop1(x, edge_index)
        x = self.lin3(x)
        return x
