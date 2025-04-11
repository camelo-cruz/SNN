# Key changes:
#  - hidden_size=100
#  - epochs=10
#  - lr=0.005
#  - T=100
#  - tau=30

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Network import Network
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

class ContrastiveNetwork(Network):
    def __init__(self, lr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def free_phase(self, image):
        self.clear_neurons()
        return self.forward(image)

    def clamped_phase(self, image, target):
        self.clear_neurons()
        # Hard clamp example (unchanged):
        clamped_hidden_activity = F.one_hot(target, num_classes=10).float() * self.T
        clamped_hidden_activity = clamped_hidden_activity.view(-1)
        return clamped_hidden_activity

    def contrastive_update(self, free_act, clamped_act):
        encoded_image = self.input_layer.encoded_image.sum(dim=1).view(-1)
        delta_w = self.lr * torch.outer(encoded_image, (clamped_act - free_act))
        self.synapse_input_hidden.w += delta_w

        # Optional: weight normalization step
        #with torch.no_grad():
            #max_norm = 2.0
            #row_norms = torch.norm(self.synapse_input_hidden.w, dim=1)
            # Clip each row if it exceeds max_norm
            #for i, row_norm in enumerate(row_norms):
            #    if row_norm > max_norm:
            #        self.synapse_input_hidden.w[i, :] *= (max_norm / row_norm)

def main():
    wandb.init(
            # set the wandb project where this run will be logged
            project="chl_training",

            # optional: give your run a short name
            name="checking new config",

            # optional: add a description of the run
            notes="checking biological params with no homeostasis and no weight normalization",

            # track hyperparameters and run metadata
            
            config={'num_train_data': 60000,
                    'num_test_data': 10000,
                    'report_interval': 1000,
                    'num_epochs': 1,
                    'T':100,
                    'hidden_size':10,
                    'tau':30,
                    'lr':0.005,
                    'R':1.0,
                    'scale':5,
                    'dt':1,
                    'V_rest':0.0,
                    'theta':0.5,
                    'refractory_period':5,
                    'tau_theta':150,
                    'theta_increment': 30,
                    'homeostasis': False,
                    'device':'cpu'}
        )
    
    config = wandb.config
    
    epochs = config['num_epochs']
    num_train_samples = config['num_train_data']
    num_test_samples = config['num_test_data']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = ContrastiveNetwork(
        lr=config['lr'],         # (1) learning rate (dimensionless)
        T=config['T'],            # (2) total simulation timesteps
        input_size=config['input_size'],   # (3) input dimensionality (MNIST)
        hidden_size=config['hidden_size'],   # (4) hidden layer size
        tau=config['tau'],           # (5) membrane time constant (ms)
        R=config['R'],              # (6) membrane resistance (normalized)
        scale=config['scale'],          # (7) input scaling factor (dimensionless)
        dt=config['dt'],             # (8) timestep (ms)
        V_rest=config['V_rest'],         # (9) resting potential (normalized)
        theta=config['theta'],        # (10) firing threshold (normalized)
        tau_theta=config['tau_theta'],    # (11) time constant for theta (ms)
        theta_increment=config['theta_increment'],  # (12) increment for theta (normalized)
        refractory_period=config['refractory_period'],  # (11) refractory period (ms)
    )
    net.homeostasis = config['homeostasis']

    print(f"Network initialized with parameters: {net}", flush=True)

    initial_w = net.synapse_input_hidden.w.clone()
    print('Initial weights norm:', torch.norm(initial_w).item())

    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{epochs}")
        for idx, (image, label) in enumerate(tqdm(train_loader, desc=f"Training for epoch {epoch+1}")):
            if idx >= num_train_samples:
                break

            free_act = net.free_phase(image)
            clamped_act = net.clamped_phase(image, label)
            net.contrastive_update(free_act, clamped_act)

            if idx % 10000 == 0:
                print(f"Step {idx}: Free activity: {free_act}, Clamped activity: {clamped_act}")
                print(f'Weights norm in {idx}:', torch.norm(net.synapse_input_hidden.w).item())

                # --- Testing loop ---
                y_true = []
                y_pred = []
                correct = 0
                total = 0

                for idx, (image, label) in enumerate(tqdm(test_loader, desc=f"Testing {idx}")):

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
                wandb.log({'accuracy': accuracy})
                print(f"Current Accuracy: {accuracy:.2%}", flush=True)
    
    # Generate and display the confusion matrix.
    labels = sorted(torch.unique(torch.tensor(train_loader.dataset.targets)).tolist())
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:\n", conf_matrix, flush=True)

    os.makedirs('plots/train_test', exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('plots/train_test/confusion_matrix.png')
    print("Confusion Matrix saved to plots/train_test/confusion_matrix.png", flush=True)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
