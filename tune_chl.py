import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np

# Import your training/testing functions and Network class
from main import train, test, assign_neuron_class_mapping
from ContrastiveNetwork import ContrastiveNetwork

# --- 1. Prepare datasets and DataLoaders ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# (Optionally limit dataset size for testing purposes)
train_dataset.data, train_dataset.targets = train_dataset.data[:60000], train_dataset.targets[:60000]
test_dataset.data, test_dataset.targets = test_dataset.data[:10000], test_dataset.targets[:10000]

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# --- 2. Define the objective function ---
def objective(config):
    # Convert wandb.config to a regular dictionary
    params = dict(config)
    homeostasis = params['homeostasis']
    del params['homeostasis']
    net = ContrastiveNetwork(**params)
    net.homeostasis = homeostasis
    
    epochs = 1

    initial_w = net.synapse_input_hidden.w.clone()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}", flush=True)
        for idx, (image, label) in enumerate(tqdm(train_loader, desc=f"Training for epoch {epoch+1}")):

            free_act = net.free_phase(image)
            clamped_act = net.clamped_phase(image, label)
            net.contrastive_update(free_act, clamped_act)

        # Check weight changes using L2 norm difference.
        last_w = net.synapse_input_hidden.w
        weight_change_norm = torch.norm(last_w - initial_w).item()
        print('Weight change (L2 norm):', weight_change_norm, flush=True)

        # --- Testing loop ---
        y_true = []
        y_pred = []
        correct = 0
        total = 0

        for idx, (image, label) in enumerate(tqdm(test_loader, desc=f"Testing for epoch {epoch+1}")):

            outs = net.forward(image)
            # For batch size 1 with output shape [num_classes], use dim=0.
            predicted_index = torch.argmax(outs, dim=0)
            predicted = predicted_index.item()
            label_val = label.item()

            y_true.append(label_val)
            y_pred.append(predicted)

            if predicted == label_val:
                correct += 1
            total += 1

        accuracy = correct / total
        print(f"Current Accuracy: {accuracy:.2%}", flush=True)
    
    return accuracy

# --- 3. Define the main function that initializes wandb and runs the objective ---
def main():
    wandb.init(project="parameter-tuning-CHL")
    # Get the configuration from wandb
    config = wandb.config
    accuracy = objective(config)
    wandb.log({"accuracy": accuracy})

# --- 4. Define the wandb sweep configuration ---
sweep_configuration = {
    "name": 'random for CHL',
    "method": "random",  # or "random" if you prefer random search first
    "notes" : "trying biological parameters",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
         "lr": {
            "min": 1e-4,      
            "max": 5e-3       
        },
        
        "decay": {
            "min": 0.0,
            "max": 0.1
        },

        "T": {"values": [100, 150, 200]},

        "spikes": {"min": 0.05,
            "max": 1.0},

        "hidden_size": {"value": 10},
         
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
            "value": 0.5,
        },
         
        # Refractory period in discrete steps
        "refractory_period": {
            "values": [3, 5, 7, 10]
        },

        "homeostasis": {
            "value": False,
        },

        #"tau_theta": {
        #    "values": [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200]
        #},

        #"theta_increment": {
        #    "values": [1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]
        #},
        
         
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
