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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- 2. Define the objective function ---
def objective(config):
    # Convert wandb.config to a regular dictionary
    params = dict(config)
    
    net = ContrastiveNetwork(**params)
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
    'name': 'bayes for CHL',
    "method": "bayes",  # or "random" if you prefer random search first
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
         "lr": {
            "distribution": "log_uniform_values", 
            "min": 1e-4, 
            "max": 1e-2},

        "T": {"values": [50, 80, 100, 150]},

        "hidden_size": {"value": 10},
         
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
        "scale": {"values": [1, 3, 5, 7, 10]},
         
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
