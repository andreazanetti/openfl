import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import LayerNorm

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path    = './data/'
dataset = MoleculeNet(path, name='HIV').shuffle() #MoleculeNet(path, name='PCBA').shuffle()
ldata = 4000
dataset = dataset[:ldata] # reducing for test and local execution

N             = len(dataset) // 10
val_dataset   = dataset[:N]
test_dataset  = dataset[N:2 * N]
train_dataset = dataset[2 * N:]

BS = 16# 32 # 128
epochs = 5
log_interval = 10
train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1) #BS
test_loader  = DataLoader(test_dataset, batch_size=1) #BS

# tensorboard
from torch.utils.tensorboard import SummaryWriter
import random
unique_id = random.randint(10,200000)
writer = SummaryWriter('./logs/experimental_interface_original_{}_BS{}_ldata{}'.format(unique_id, BS, ldata), flush_secs=5)

def write_metric(node_name, task_name, metric_name, metric, round_number):
    writer.add_scalar("{}/{}/{}".format(node_name, task_name, metric_name),
                      metric, round_number)


class GNN(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int = 512,
        num_layers: int = 3):
        super(GNN, self).__init__()

        # Initial layer to map features -> hidden
        self.lin0 = nn.Linear(in_features, hidden)

        # Define GNN layers
        self.convs = nn.ModuleList([GCNConv(hidden * 2, hidden) for _ in range(num_layers)])

        # Define norm layers
        self.norms = nn.ModuleList([LayerNorm(hidden) for _ in range(num_layers)])

        # Final linear layers
        self.lin1 = nn.Linear(hidden * 2, hidden * 4)
        self.lin2 = nn.Linear(hidden * 4, hidden * 4)
        self.lin3 = nn.Linear(hidden * 4, out_features)


    def forward(self, x, edge_index, edge_attr, batch):

        # First feature transform layer
        x = F.leaky_relu(self.lin0(x))

        for i, conv in enumerate(self.convs):

            x_glob = torch.index_select(gap(x, batch), 0, batch)   # Virtual global node
            x = torch.cat((x, x_glob), dim=1)
            x = conv(x=x, edge_index=edge_index)
            x = self.norms[i](x, batch)
            x = F.leaky_relu(x)
            if i == 0:
                x_pooled = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # Global pooling after each layer
            else:
                x_pooled = x_pooled + torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.leaky_relu(self.lin1(x_pooled))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x) # shape: (batch_size, out_features)

        return x


model = GNN(hidden=32,
            in_features=dataset.num_features,
            num_layers=3,
            out_features=dataset[0].y.numel()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    total_loss = 0
    for idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
        loss = (out - data.y).abs().mean() 
        # print("loss for batch is: ", loss)
        # print("data.y.mean() is {} and out.mean() is {}".format(data.y.mean(), out.mean()))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        if idx % log_interval == 0:
            write_metric("Original", "train", "total_loss", loss.item(), epoch * len(train_loader) + idx)
            print("Loss at epoch {} batch {} is: {}".format(epoch, idx, loss.item()))

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    total_error = 0
    # correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    #     if (out.data - data.y).abs() < 0.5: #BS eval = 1
    #         correct +=1
    # accuracy = correct / len(test_loader.dataset)
    # print("###################### Evaluation accuracy is: ", accuracy)
    return total_error / len(loader.dataset)


for epoch in range(0, epochs): #301):
    loss = train(epoch)
    val_mae = test(val_loader)
    test_mae = test(test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')