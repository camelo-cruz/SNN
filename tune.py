import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

# Import your training/testing functions and Network class
from main import train, test, assign_neuron_class_mapping
from Network import Network

# --- 1. Prepare datasets and DataLoaders ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# (Optionally limit dataset size for testing purposes)
train_dataset.data, train_dataset.targets = train_dataset.data[:40000], train_dataset.targets[:40000]
test_dataset.data, test_dataset.targets = test_dataset.data[:10000], test_dataset.targets[:10000]

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- 2. Define the objective function ---
def objective(config):
    # Convert wandb.config to a regular dictionary
    params = dict(config)
    
    # Compute dependent parameter A_minus
    params['A_minus'] = params['A_plus'] * params['A_ratio']

    params.pop('A_ratio', None)
    
    network = Network(**params)
    
    # Proceed with training and evaluation
    train(network, train_loader, 1000)
    assign_neuron_class_mapping(network, train_loader, 1000)
    accuracy = test(network, test_loader, len(test_dataset))
    
    return accuracy

# --- 3. Define the main function that initializes wandb and runs the objective ---
def main():
    wandb.init(project="parameter-tuning-sweep")
    # Get the configuration from wandb
    config = wandb.config
    accuracy = objective(config)
    wandb.log({"accuracy": accuracy})

# --- 4. Define the wandb sweep configuration ---
sweep_configuration = {
     'name': 'bayes with different weight initialization no clamping',
     "method": "bayes",  # or "random" if you prefer random search first
     "metric": {"goal": "maximize", "name": "accuracy"},
     "parameters": {
         "T": {"value": 50},
         "hidden_size": {"values": [100, 200, 300, 400, 500]},
         
         # Tau (membrane time constant) in fairly spaced discrete steps:
         "tau": {
             "values": [30, 40, 50, 70, 100, 130, 150, 200]
         },
         
         # Membrane resistance
         "R": {
             "min": 0.1,
             "max": 1.0
         },
         
         # Scale factor for weights (if relevant)
         "scale": {"values": [3, 5, 7, 10]},
         
         # Simulation timestep
         "dt": {"value": 1},
         
         # Resting potential range
         "V_rest": {
             "min": 0.1,
             "max": 1.0
         },
         
         # Baseline threshold
         "theta": {
             "min": 1,
             "max": 10,
             "distribution": "int_uniform"
         },
         
         # Refractory period in discrete steps
         "refractory_period": {
             "values": [3, 5, 7, 9, 11, 13, 15, 20]
         },
         
         # Adaptation time constant of threshold
         "tau_theta": {
             "values": [10, 20, 30, 50, 70, 100, 130, 150, 200]
         },
         
         # Threshold increment upon spike
         "theta_increment": {
             "values": [10, 15, 20, 30, 40, 50, 60, 70, 100, 200]
         },
         
         # STDP LTP magnitude
         "A_plus": {
             "min": 1e-6,
             "max": 1e-4
         },
         
         # Ratio that determines how A_minus is set relative to A_plus
         "A_ratio": {
             "min": 1.0,
             "max": 2.0
         },
         
         # STDP time constant (decay of pre/post traces)
         "tau_stdp": {
             "values": [10, 20, 30, 50, 70, 100, 130, 150, 200]
         },
         
         # Weight bounds
         "max_w": {"value": 255},
         "min_w": {"value": 0},
         
         # Device
         "device": {"value": "cpu"},
     },
 }

# --- 5. Initialize the sweep and run the wandb agent ---
wandb.login()
sweep_id = wandb.sweep(sweep=sweep_configuration, project="parameter-tuning-sweep", entity="camelo-cruz-university-of-potsdam")
wandb.agent(sweep_id, 
            function=main, 
            count=1000,
            entity="camelo-cruz-university-of-potsdam",  # or your org name if it's a team account
            project="parameter-tuning-sweep")
