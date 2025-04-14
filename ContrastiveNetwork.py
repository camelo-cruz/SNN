import os
import wandb
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Neuron import CurrentLIF
from Receptive_Field import ReceptiveField
from typeguard import typechecked
from Synapse import Synapse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

@typechecked
class ContrastiveNetwork:
    def __init__(self,
                 # CHL dynamics
                 lr: float,
                 gamma: float,
                 decay: float, 
                 spikes: float,
                 T: int,
                 layer_sizes: list,  # List of sizes, e.g. [input_size, hidden1_size, hidden2_size, ..., output_size]
                 # Neuron parameters
                 tau: float,
                 R: float,
                 scale: float,
                 dt: float,
                 V_rest: float,
                 theta: float,
                 refractory_period: int,
                 record_history: bool = False,
                 device: str = 'cpu',
                 ):
        self.lr = lr
        self.gamma = gamma
        self.decay = decay
        self.spikes = spikes
        self.T = T
        self.layer_sizes = layer_sizes  # Now includes the input, hidden, and output dimensions
        self.L = len(layer_sizes)       # Total number of layers
        self.tau = tau
        self.R = R
        self.scale = scale
        self.dt = dt
        self.V_rest = V_rest
        self.theta = theta
        self.refractory_period = refractory_period

        # Additional parameters
        self.record_history = record_history
        self.device = device

        # Initialize neurons and synapses
        self.create_neurons()
        self.create_synapses()
    
    def __repr__(self):
        return (f"ContrastiveNetwork(T={self.T}, layer_sizes={self.layer_sizes}, lr={self.lr}, gamma={self.gamma}, "
                f"decay={self.decay}, spikes={self.spikes}, tau={self.tau}, R={self.R}, scale={self.scale}, dt={self.dt}, "
                f"V_rest={self.V_rest}, theta={self.theta}, refractory_period={self.refractory_period})")
    
    def clear_neurons(self):
        """
        Reset the internal state of all neurons.
        """
        for i, layer in enumerate(self.layers):
            if i > 0: #doesnt apply for input layer
                layer.reset()
    
    def create_neurons(self):
        """
        Initialize the input and subsequent (hidden/output) layers.
        """
        self.layers = []

        # Create the input layer using ReceptiveField.
        input_size = self.layer_sizes[0]
        self.input_layer = ReceptiveField(
            input_size=input_size,
            record_history=self.record_history,
            device=self.device
        )
        self.layers.append(self.input_layer)
        print(f"Initialized input layer of size {self.input_layer.batch_size}", flush=True)
        
        # Create subsequent layers (hidden and/or output) using CurrentLIF.
        for size in self.layer_sizes[1:]:
            layer = CurrentLIF(
                V_rest=self.V_rest,
                theta=self.theta,
                scale=self.scale,
                refractory_period=self.refractory_period,
                record_history=self.record_history,
                batch_size=size,
                tau=self.tau,
                R=self.R,
                dt=self.dt,
                device=self.device
            )
            self.layers.append(layer)
            print(f"Initialized layer with size {size}", flush=True)
    
    def create_synapses(self):
        """
        Create feedforward synapses connecting adjacent layers and feedback synapses between internal layers.
        """
        self.feedforward_synapses = []
        self.W = []
        # Create weight matrices and feedforward synapses for each pair of consecutive layers.
        for i in range(1, self.L):
            # Weight matrix shape: (previous_layer_size, current_layer_size)
            w = torch.normal(
                mean=0.0,
                std=0.1,
                size=(self.layer_sizes[i-1], self.layer_sizes[i]),
                device=self.device
            )
            self.W.append(w)
            synapse = Synapse(
                pre_neuron=self.layers[i-1],
                post_neuron=self.layers[i],
                w=w
            )
            self.feedforward_synapses.append(synapse)
        
        print(f"Initialized {len(self.feedforward_synapses)} feedforward synapses", flush=True)
        for i, synapse in enumerate(self.feedforward_synapses):
            print(f"Feedforward synapse {i} weight norm: {torch.norm(synapse.w).item()}", flush=True)
        
        self.feedback_synapses = []
        if self.L > 2:
            for i in range(2, self.L):
                feedback_w = self.W[i-1].t()
                synapse = Synapse(
                    pre_neuron=self.layers[i],
                    post_neuron=self.layers[i-1],
                    w=feedback_w
                )
                self.feedback_synapses.append(synapse)
            print(f"Initialized {len(self.feedforward_synapses)} feedforward synapses and {len(self.feedback_synapses)} feedback synapses", flush=True)
    
    def free_phase(self, image):
        """
        Run the free phase dynamics.
        """
        self.input_layer.compute_potential(image, encoding='isi', time_steps=self.T)
        bs = image.shape[0]
        # x_free holds spiking activity for each layer: index 0 for input, 1..L-1 for each subsequent layer.
        x0 = self.input_layer.encoded_image.sum(dim=1).view(bs, -1).int()
        x_free = [x0]
        for layer in self.layers[1:]:
            x_free.append(torch.zeros((bs, layer.batch_size), dtype=torch.int32, device=self.device))
        
        for t in range(self.T):
            self.input_layer.fire(t)
            # Feedforward pass.
            for synapse in self.feedforward_synapses:
                synapse.transmit()
            # Feedback pass.
            for synapse in self.feedback_synapses:
                synapse.transmit()
            # Update each layer (except input) and accumulate spiking activity.
            for i, layer in enumerate(self.layers[1:], start=1):
                layer.update()
                x_free[i] += layer.spiked
        
        self.clear_neurons()
        return x_free

    def clamped_phase(self, image, target):
        """
        Run the clamped phase dynamics.
        The target is applied at the last layer.
        """
        # Keep the one-hot tensor shape as (batch_size, num_classes)
        one_hot_target = F.one_hot(target, num_classes=self.layer_sizes[-1]).int().view(-1)  # shape: (bs, num_classes)
        
        # Compute the input potential.
        self.input_layer.compute_potential(image, encoding='isi', time_steps=self.T)
        bs = image.shape[0]
        
        # Compute the input layer activity.
        x0 = self.input_layer.encoded_image.sum(dim=1).view(bs, -1).int()
        x_clamped = [x0]

        # Initialize the activity for subsequent layers.
        for layer in self.layers[1:]:
            x_clamped.append(torch.zeros((bs, layer.batch_size), dtype=torch.int32, device=self.device))
        
        # Process T time steps, updating only the hidden layers.
        for t in range(self.T):
            self.input_layer.fire(t)
            scaled_target = torch.bernoulli(one_hot_target.float() * self.spikes).int()
            self.layers[-1].spiked = scaled_target
            x_clamped[-1] += self.layers[-1].spiked

            for synapse in self.feedforward_synapses:
                synapse.transmit()
            for synapse in self.feedback_synapses:
                synapse.transmit()
                
            # Update only the hidden layers (skip input and output).
            for i, layer in enumerate(self.layers[1:-1], start=1):
                layer.update()
                x_clamped[i] += layer.spiked
                
        self.clear_neurons()
        return x_clamped



    def contrastive_update(self, x_free, x_clamped):
        """
        Update feedforward weights using the contrastive difference between clamped and free phases.
        """
        for k, w in enumerate(self.W):
            coeff = self.lr * self.gamma**(k-(len(self.layer_sizes)-1))
            w += coeff * (x_clamped[k].T @ x_clamped[k+1] - x_free[k].T @ x_free[k+1])
        # Update the synapses with the new weights.
        for i, synapse in enumerate(self.feedforward_synapses):
            synapse.w = self.W[i]
        for i, synapse in enumerate(self.feedback_synapses):
            # Feedback synapse from layer i+1 to i uses the transpose of the corresponding feedforward weight.
            synapse.w = self.W[i+1].t()

#--- Import data ---
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

def simulation():
    net = ContrastiveNetwork(
        lr=0.0000006,
        gamma=0.1,
        decay=0.0004,
        spikes=0.10,
        T=50,
        input_size=784,
        hidden_size=[100, 10],
        tau=30,
        R=1.0,
        scale=3.6,
        dt=1.0,
        V_rest=0.0,
        theta=0.5,
        refractory_period=5,
    )

    print(f"Network initialized with parameters: {net}", flush=True)
    for idx, (image, label) in enumerate(tqdm(train_loader, desc="Training")):
        if idx >= 2:
            break
        # Free phase
        free_act = net.free_phase(image)
        # Clamped phase
        clamped_act = net.clamped_phase(image, label)
        # Contrastive update
        net.contrastive_update(free_act, clamped_act)
        print(f"label {label}: Free activity: {free_act}, Clamped activity: {clamped_act}")

def train(network, train_loader, num_train_samples, epochs, test_loader = None, num_test_samples = None):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for idx, (image, label) in enumerate(tqdm(train_loader, desc=f"Training for epoch {epoch+1}")):
            if idx >= num_train_samples:
                break

            free_act = network.free_phase(image)
            clamped_act = network.clamped_phase(image, label)
            network.contrastive_update(free_act, clamped_act)

            if idx % 10000 == 0:
                print(f"Step {idx} label {label}: Free activity: {free_act[-1]}, Clamped activity: {clamped_act[-1]}")
                if test_loader:
                    results = test(network, test_loader, num_test_samples)
                    wandb.log({'accuracy': results['accuracy']})
                    for i, synapse in enumerate(network.feedforward_synapses):
                        print(f'Weights norm in {idx}:', torch.norm(synapse.w).item())

                
def test(network, test_loader, num_test_samples):
    # --- Testing loop ---
    y_true = []
    y_pred = []
    correct = 0
    total = 0

    for idx, (image, label) in enumerate(tqdm(test_loader, desc=f"Testing")):
        if idx >= num_test_samples:
            break
        predicted = torch.argmax(network.free_phase(image)[-1], dim=1)
        predicted = predicted.item()
        label_val = label.item()

        y_true.append(label_val)
        y_pred.append(predicted)

        if predicted == label_val:
            correct += 1
        total += 1
    accuracy = correct / total

    print(f"Accuracy: {accuracy:.2%}", flush=True)

    return {'accuracy': accuracy, 'y_true': y_true, 'y_pred': y_pred}


def main():
    wandb.init(
            # set the wandb project where this run will be logged
            project="chl_training",

            # optional: add a description of the run
            notes="checking biological params and weight normalization",

            # track hyperparameters and run metadata
            
            config={'num_train_data': 60000,
                    'num_test_data': 10000,
                    'report_interval': 1000,
                    'num_epochs': 2,
                    'T':50,
                    'layer_sizes':[784, 10],
                    'tau':30,
                    'lr':0.01,
                    'gamma':0.1,
                    'spikes' : 1.0,
                    'decay': 0.0004,
                    'R':1.0,
                    'scale':3.6,
                    'dt':1.0,
                    'V_rest':0.0,
                    'theta':0.5,
                    'refractory_period':5,
                    'device':'cpu'}
        )
    

    config = wandb.config
    
    epochs = config['num_epochs']
    num_train_samples = config['num_train_data']
    num_test_samples = config['num_test_data']


    net = ContrastiveNetwork(
        lr=config['lr'],
        gamma=config['gamma'],
        decay=config['decay'],
        spikes=config['spikes'],
        T=config['T'],
        layer_sizes=config['layer_sizes'],
        tau=config['tau'],
        R=config['R'],
        scale=config['scale'],
        dt=config['dt'],
        V_rest=config['V_rest'],
        theta=config['theta'],
        refractory_period=config['refractory_period'],
    )

    print(f"Network initialized with parameters: {net}", flush=True)
    
    train(net, train_loader, num_train_samples=num_train_samples, epochs=epochs, test_loader=test_loader, num_test_samples=100)
    results = test(net, test_loader, num_test_samples=num_test_samples)
    accuracy = results['accuracy']
    y_true = results['y_true']
    y_pred = results['y_pred']

    wandb.log({'accuracy': accuracy})
    print(f"Final Accuracy: {accuracy:.2%}", flush=True)
    
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

    wandb.finish()

if __name__ == "__main__":
    main()