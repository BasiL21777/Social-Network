# improved_gcn.py
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class ImprovedGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def enhance_features(data):
    degrees = torch.zeros(data.num_nodes, 1)
    for i in range(data.num_nodes):
        degrees[i] = (data.edge_index[0] == i).sum()
    enhanced_features = torch.cat([data.x, degrees], dim=1)
    enhanced_features = enhanced_features / enhanced_features.sum(1, keepdim=True).clamp(min=1)
    return enhanced_features

def create_balanced_masks(data, train_ratio=0.7):
    num_nodes = data.num_nodes
    class_0_indices = (data.y == 0).nonzero().squeeze()
    class_1_indices = (data.y == 1).nonzero().squeeze()

    train_indices_0 = class_0_indices[:len(class_0_indices)//2]
    train_indices_1 = class_1_indices[:len(class_1_indices)//2]

    train_indices = torch.cat([train_indices_0, train_indices_1])
    test_indices = torch.tensor([i for i in range(num_nodes) if i not in train_indices])

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return train_mask, test_mask

def prepare_data():
    # Create graph
    G = nx.Graph()
    edges = [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)]
    G.add_edges_from(edges)

    # One-hot node features
    num_nodes = 6
    features = torch.eye(num_nodes, dtype=torch.float)

    # Node labels
    labels = torch.tensor([0,0,0, 1,1,1], dtype=torch.long)

    # Convert to PyG Data
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    data = Data(x=features, edge_index=edge_index, y=labels)

    # Enhance features and create masks
    data.x = enhance_features(data)
    data.train_mask, data.test_mask = create_balanced_masks(data)

    return data

def train_model(model, data, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    early_stopper = EarlyStopper(patience=20)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if early_stopper.early_stop(loss):
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data).argmax(dim=1)
                acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

    return model

if __name__ == "__main__":
    print("=== Running Improved GCN ===")

    # Prepare data
    data = prepare_data()
    print(f"Data prepared: {data.num_nodes} nodes, {data.train_mask.sum()} train, {data.test_mask.sum()} test")

    # Test Improved GCN
    print("\n=== Testing Improved GCN ===")
    model_gcn = ImprovedGCN(num_features=data.num_node_features, num_classes=2, hidden_dim=16)
    model_gcn = train_model(model_gcn, data)

    model_gcn.eval()
    with torch.no_grad():
        pred_gcn = model_gcn(data).argmax(dim=1)
        acc_gcn = (pred_gcn[data.test_mask] == data.y[data.test_mask]).float().mean()
    print(f'Improved GCN Final Test Accuracy: {acc_gcn:.4f}')

    # Test GraphSAGE
    print("\n=== Testing GraphSAGE ===")
    model_sage = GraphSAGE(num_features=data.num_node_features, num_classes=2, hidden_dim=16)
    model_sage = train_model(model_sage, data)

    model_sage.eval()
    with torch.no_grad():
        pred_sage = model_sage(data).argmax(dim=1)
        acc_sage = (pred_sage[data.test_mask] == data.y[data.test_mask]).float().mean()
    print(f'GraphSAGE Final Test Accuracy: {acc_sage:.4f}')

    print(f"\n=== Final Results ===")
    print(f"Improved GCN Accuracy: {acc_gcn:.4f}")
    print(f"GraphSAGE Accuracy: {acc_sage:.4f}")
