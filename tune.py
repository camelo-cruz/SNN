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
    'name': 'random with corrected stdp',
    "method": "bayes",  # or "random" if you prefer random search first
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "T": {"value": 100},
        "hidden_size": {"values": [200, 500, 1000]},
        
        # Tau (membrane time constant) in fairly spaced discrete steps:
        "tau": {
            "values": [30, 50, 70, 100, 130]
        },
        
        # Membrane resistance
        "R": {
            "min": 0.4,
            "max": 1.0
        },
        
        # Scale factor for weights (if relevant)
        "scale": {"value": 1},
        
        # Simulation timestep
        "dt": {"value": 1},
        
        # Resting potential range
        "V_rest": {
            "min": 0.2,
            "max": 0.8
        },
        
        # Baseline threshold
        "theta": {
            "min": 2,
            "max": 8,
            "distribution": "int_uniform"
        },
        
        # Refractory period in discrete steps
        "refrac": {
            "values": [5, 7, 9, 11, 13]
        },
        
        # Adaptation time constant of threshold
        "tau_theta": {
            "values": [30, 50, 70, 100]
        },
        
        # Threshold increment upon spike
        "theta_increment": {
            "values": [10, 20, 30, 50]
        },
        
        # STDP LTP magnitude
        "A_plus": {
            "min": 1e-5,
            "max": 5e-5
        },
        
        # Ratio that determines how A_minus is set relative to A_plus
        "A_ratio": {
            "min": 1.0,
            "max": 1.5
        },
        
        # STDP time constant (decay of pre/post traces)
        "tau_stdp": {
            "values": [30, 50, 70, 100]
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
