import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np

# Import your training/testing functions and Network class
from ContrastiveNetwork import ContrastiveNetwork, train, test

# --- 1. Prepare datasets and DataLoaders ---
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# --- 2. Define the objective function ---
def objective(config):
    # Convert wandb.config to a regular dictionary
    params = dict(config)
    net = ContrastiveNetwork(**params)
    
    epochs = 1
    num_train_samples = 40000
    num_test_samples = 10000

    train(net, train_loader, num_train_samples=num_train_samples, epochs=epochs)
    results = test(net, test_loader, num_test_samples=num_test_samples)
    
    return results['accuracy']

# --- 3. Define the main function that initializes wandb and runs the objective ---
def main():
    wandb.init(project="parameter-tuning-CHL")
    # Get the configuration from wandb
    config = wandb.config
    accuracy = objective(config)
    wandb.log({"accuracy": accuracy})

# --- 4. Define the wandb sweep configuration ---
sweep_configuration = {
    "name": 'bayes for CHL',
    "method": "random",  # or "random" if you prefer random search first
    "notes" : "trying biological parameters",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
         "lr": {
            "min": 1e-7,
            "max": 0.1     
        },

        "gamma": {
            "min": 1e-7,      
            "max": 0.1     
        },

        "input_size": {
            "value": 784
        },
        
        "decay": {
            "min": 0.0,
            "max": 0.1
        },

        "T": {"values": [100, 150, 200]},

        "spikes": {"min": 0.05,
            "max": 1.0},

        "hidden_size": {"value": [10]},
         
        # Tau (membrane time constant) in fairly spaced discrete steps:
        "tau": {
            "values": [30, 40, 50, 60, 70, 80, 90, 100]
        },
         
        # Membrane resistance
        "R": {
            "value": 1.0,
        },
         
        # Scale factor for weights (if relevant)
        "scale": {"min": 1.0,
                 "max": 10.0
                },
         
        # Simulation timestep
        "dt": {"value": 1},
         
        # Resting potential range
        "V_rest": {
            "value": 0.0,
        },
         
        # Baseline threshold
        "theta": {
            "values": [0.5, 1.0, 1.5, 2.0]
        },
         
        # Refractory period in discrete steps
        "refractory_period": {
            "values": [3, 5, 7, 10]
        },
         
        # Device
        "device": {"value": "cpu"},
     },
 }

# --- 5. Initialize the sweep and run the wandb agent ---
wandb.login()
sweep_id = wandb.sweep(sweep=sweep_configuration, project="parameter-tuning-CHL", entity="camelo-cruz-university-of-potsdam")
wandb.agent(sweep_id, 
            function=main, 
            count=1000,
            entity="camelo-cruz-university-of-potsdam",  # or your org name if it's a team account
            project="parameter-tuning-CHL")
