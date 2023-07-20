import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy

#openFL imports
from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator

# GNN imports
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import LayerNorm
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

# reproducibility...
# torch.backends.cudnn.enabled = False
def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
set_seed(10)
random_seed = 1
torch.manual_seed(random_seed)

# device and data path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path    = './data/'
dataset = MoleculeNet(path, name='HIV').shuffle() #MoleculeNet(path, name='PCBA').shuffle()
ldata = 4000
dataset = dataset[:ldata] # reducing for test and local execution

# Dataset definitions
N             = len(dataset) // 10
val_dataset   = dataset[:N]
test_dataset  = dataset[N:2 * N]
train_dataset = dataset[2 * N:]

train_labels = train_dataset.y
valid_labels = val_dataset.y

# hyperparams
BS = 16# 32 # 128
n_epochs = 5
momentum = 0.5
log_interval = 10

# tensorboard
from torch.utils.tensorboard import SummaryWriter
import random
unique_id = random.randint(10,2000000) # random 'unique' id for tensorboard clarity
writer = SummaryWriter('./logs/experimental_interface_openFL_{}_BS{}_ldata{}'.format(unique_id, BS, ldata), flush_secs=5)

def write_metric(node_name, task_name, metric_name, metric, round_number):
    writer.add_scalar("{}/{}/{}".format(node_name, task_name, metric_name),
                      metric, round_number)
    
# Model defintion
class GNN(torch.nn.Module):
    def __init__(
        self,
        in_features: int = 32,
        out_features: int = 1, # it should be dataset[0].y.numel()).to(device)
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

# loss function definition 
def myloss(output, target):
    return (output - target).abs().mean() 

# evaluation/test function
def inference(network, test_loader):
    network.eval()
    total_error = 0
    correct = 0
    for data in test_loader:
        data = data.to(device)
        out = network(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
        total_error += (out - data.y).abs().sum().item()
        correct += ((out.data - data.y).abs() < 0.5).sum() # binary classification
    accuracy = correct / len(test_loader.dataset)
    print("###################### Eval/Test accuracy is: ", accuracy)
    return total_error / len(test_loader.dataset)

# OpenFL: Averaging all parameter vectors
def FedAvg(models): #AZ review this is correct and does not introduce a blocker to training!!!!!!
    new_model = models[0]
    state_dict_keys = new_model.state_dict().keys()
    model_dict_list = [mdl.state_dict() for mdl in models]
    state_dict = new_model.state_dict()
    num_models = len(models)
    for key in state_dict_keys:
        state_dict[key] = torch.sum(
            torch.stack(
            [mdldict[key]/num_models for mdldict in model_dict_list]), dim=0)
    new_model.load_state_dict(state_dict)
    return new_model

#OpenFL: definition of compute flow
class FederatedFlow(FLSpec):
    def __init__(self, model = None, optimizer = None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            self.optimizer = optimizer #optimizer(params=self.model.parameters()) # 
            #self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.model = Net() #assuming Net() is defined somewhere else...
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                   momentum=momentum)
        self.rounds = rounds
        self.aggr_training_value_list = []

    @aggregator
    def start(self):
        print(f'Performing initialization for model')
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        self.next(self.aggregated_model_validation,foreach='collaborators',exclude=['private'])

    @collaborator
    def aggregated_model_validation(self):
        print(f'########################## Performing aggregated model validation for collaborator {self.input}')
        self.agg_validation_score = inference(self.model, self.test_loader)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)

    @collaborator
    def train(self):
        self.model.train()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
        #                            momentum=momentum)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) 
        # NOTE: optimizer seems to be defined here to be able to act on collaborator parameters
        total_loss = 0
        # print("Collaborator self.model params example BEFORE a round of training:\n ", self.model.state_dict()["lin0.weight"])

        ll = len(self.train_loader)
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(device)
            self.optimizer.zero_grad()
            out = self.model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
            loss = (out - data.y).abs().mean() 
            # print("loss for batch is: ", loss)
            # print("data.y.mean() is {} and out.mean() is {}".format(data.y.mean(), out.mean()))
            loss.backward()
            total_loss += loss.item() #* data.num_graphs
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(self.current_round,
                batch_idx * len(data), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))
                self.loss = loss.item()
                torch.save(self.model.state_dict(), 'model.pth')
                torch.save(self.optimizer.state_dict(), 'optimizer.pth')
                write_metric(self.input, "train", "loss", loss.item(),
                             self.current_round * ll + batch_idx)
                print("Wrote TensorBoard information for collaborator ", self.input)
        
        # print("Collaborator self.model params example AFTER a round of training:\n ", self.model.state_dict()["lin0.weight"])
        self.training_completed = True
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader)
        print(f'Doing local model validation for collaborator {self.input}: {self.local_validation_score}')
        self.next(self.join, exclude=['training_completed'])

    @aggregator
    def join(self, inputs):
        print(f'############## AGGREGATOR NOW: time to join!! for Epoch {self.current_round}')
        self.average_loss = sum(input.loss for input in inputs)/len(inputs)
        self.aggr_training_value_list.append(self.average_loss)
        self.aggregated_model_accuracy = sum(input.agg_validation_score for input in inputs)/len(inputs)
        self.local_model_accuracy = sum(input.local_validation_score for input in inputs)/len(inputs)
        print(f'Average aggregated model validation values = {self.aggregated_model_accuracy}')
        print(f'Average training loss = {self.average_loss}')
        print(f'Average local model validation values = {self.local_model_accuracy}')
        
        self.model = FedAvg([input.model for input in inputs])
        self.optimizer = [input.optimizer for input in inputs][0] # AZ what is this doing?

        write_metric("Aggregator", "train", "avg_loss", self.average_loss, self.current_round)
        print("Wrote TensorBoard information for Aggregator ")

        self.current_round += 1
        if self.current_round < self.rounds:
            self.next(self.aggregated_model_validation, foreach='collaborators', exclude=['private'])
        else:
            self.next(self.end)
        
    @aggregator
    def end(self):
        print("List of averaged training losses :\n", self.aggr_training_value_list)
        print(f'This is the end of the flow') 
        
# Setup participants
aggregator = Aggregator()
aggregator.private_attributes = {}

# Setup collaborators with private attributes
collaborator_names = ['Coll_1', 'Coll_2']
collaborators = [Collaborator(name=name) for name in collaborator_names]

for idx, collaborator in enumerate(collaborators):
    local_train = deepcopy(train_dataset)
    local_test = deepcopy(test_dataset)
    test_l = len(local_test)
    train_l = len(local_train)
    local_train = train_dataset[idx*train_l//2: (idx+1)*train_l//2] # this is ok only for 2 collaborators....
    local_test = test_dataset[idx*test_l//2: (idx+1)*test_l//2]
    collaborator.private_attributes = {
            'train_loader': DataLoader(local_train, batch_size=BS, shuffle=True),
            'test_loader': DataLoader(local_test, batch_size=BS)
    }

local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, backend='single_process')
print(f'Local runtime collaborators = {local_runtime.collaborators}')

model = GNN(hidden=32,
            in_features=dataset.num_features,
            num_layers=3,
            out_features=dataset[0].y.numel()) #.to(device)
best_model = None

# from functools import partial # tested with no success for now... optmizer still need to be defined in collaborator to act on params
# optimizer = partial(optim.Adam, lr=0.001)
optimizer = None
flflow = FederatedFlow(model, optimizer, rounds=n_epochs)
flflow.runtime = local_runtime

print("NOW CALLING .run -----------------------------------------------------------------------------------------------------")
flflow.run()
print(".run has finished-----------------------------------------------------------------------------------------------------")

print(f'\nFinal aggregated model accuracy for {flflow.rounds} rounds of training: {flflow.aggregated_model_accuracy}')


