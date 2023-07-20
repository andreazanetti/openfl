import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import openfl.native as fx
from openfl.federated import FederatedModel,FederatedDataSet
import random
import warnings
warnings.filterwarnings('ignore')


# GNN imports
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import LayerNorm
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

# device +  data specification and definition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path    = './data/'
dataset = MoleculeNet(path, name='HIV').shuffle() #MoleculeNet(path, name='PCBA').shuffle()
dataset = dataset[:1000] # reducing for test and local execution

N             = len(dataset) // 10
val_dataset   = dataset[:N]
test_dataset  = dataset[N:2 * N]
train_dataset = dataset[2 * N:]

train_labels = train_dataset.y
valid_labels = val_dataset.y

# PYG dataloaders! they return complex objects and not simple numpy arrays
BS = 32 # 128
# maybe train data loaders are not useful here.
# I need to pass the data to FederatedDataset object and there
# once appropriated, the data loaders for each collaborators are created
# train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=BS)
# test_loader  = DataLoader(test_dataset, batch_size=BS)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
set_seed(10)


#Setup default openfl workspace, logging, etc.
fx.init('torch_cnn_mnist') # why is this needed? this creates 2 collaborators in any case... why?
# but it also creates a set of folders useful later on.... maybe that is why we run it here?

# Define our dataset and model to perform federated learning on. 
# The dataset should be composed of a Numpy array
# We start with a simple fully connected model that is trained on the MNIST dataset.

# def one_hot(labels, classes):
#     return np.eye(classes)[labels]

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                         download=True, transform=transform)

# train_images,train_labels = trainset.train_data, np.array(trainset.train_labels)
# train_images = torch.from_numpy(np.expand_dims(train_images, axis=1)).float()
# train_labels = one_hot(train_labels,10)

# validset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=True, transform=transform)

# valid_images,valid_labels = validset.test_data, np.array(validset.test_labels)
# valid_images = torch.from_numpy(np.expand_dims(valid_images, axis=1)).float()
# valid_labels = one_hot(valid_labels,10)

# feature_shape = train_images.shape[1]
# classes       = 10
classes = 2

fl_data = FederatedDataSet(train_dataset, 
                           train_labels, 
                           val_dataset, 
                           valid_labels, 
                           batch_size=BS, 
                           num_classes=classes)


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
    
optimizer = lambda x: optim.Adam(x, lr=1e-4)

# def cross_entropy(output, target):
#     """Binary cross-entropy metric
#     """
#     return F.cross_entropy(input=output,target=target)

def myloss(output, target):
    return (output - target).abs().mean() 

# Here we can define metric logging function. It should has the following signature described below. 
# You can use it to write metrics to tensorboard or some another specific logging.
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs/cnn_mnist', flush_secs=5)

def write_metric(node_name, task_name, metric_name, metric, round_number):
    writer.add_scalar("{}/{}/{}".format(node_name, task_name, metric_name),
                      metric, round_number)

# Create a federated model using the pytorch class, lambda optimizer function, and loss function
# The FederatedModel object is a wrapper around your PyTorch model that makes it compatible with openfl. 
# It provides built in federated training and validation functions that we will see used below. 
# Using it's setup function, collaborator models and datasets can be automatically 
# defined for the experiment.
fl_model = FederatedModel(build_model=GNN,
                        optimizer=optimizer,
                        loss_fn=myloss,
                        data_loader=fl_data)

collaborator_models = fl_model.setup(num_collaborators=3) # here the call to FederatedModel.setup occurs!
collaborators = {'one':collaborator_models[0],'two':collaborator_models[1], 'three':collaborator_models[2]}

#Original MNIST dataset
print(f'Original training data size: {len(train_images)}')
print(f'Original validation data size: {len(valid_images)}\n')

for i,coll in enumerate(collaborator_models):
    #Collaborator one's data
    print(f'Collaborator {i} \'s training data size: {len(coll.data_loader.X_train)}')
    print(f'Collaborator {i} \'s validation data size: {len(coll.data_loader.X_valid)}\n')

# #Collaborator two's data
# print(f'Collaborator two\'s training data size: {len(collaborator_models[1].data_loader.X_train)}')
# print(f'Collaborator two\'s validation data size: {len(collaborator_models[1].data_loader.X_valid)}\n')

#Collaborator three's data
#print(f'Collaborator three\'s training data size: {len(collaborator_models[2].data_loader.X_train)}')
#print(f'Collaborator three\'s validation data size: {len(collaborator_models[2].data_loader.X_valid)}')

# We can see the current plan values by running the fx.get_plan() functio
 #Get the current values of the plan. Each of these can be overridden
import json
print(json.dumps(fx.get_plan(), indent=4, sort_keys=True))

#  #Get the current values of the plan. Each of these can be overridden
# print(fx.get_plan())

# Now we are ready to run our experiment. 
# If we want to pass in custom plan settings, we can easily do that with the override_config parameter
# Run experiment, return trained FederatedModel
final_fl_model = fx.run_experiment(collaborators, override_config={
    'aggregator.settings.rounds_to_train': 5,
    'aggregator.settings.log_metric_callback': write_metric,
})

#Save final model
final_fl_model.save_native('final_pytorch_model')