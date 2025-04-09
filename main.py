import pickle
import wandb
import torch
import os
from tqdm import tqdm
import numpy as np
from Network import Network
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def save_network(network, filename="network.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(network, f)
        print(f"Network saved to {filename}")

def load_network(filename):
        """
        Load the entire network object from a file.

        Args:
            filename (str): Path to load the network from.
        
        Returns:
            Network: The loaded spiking neural network instance.
        """
        with open(filename, 'rb') as f:
            network = pickle.load(f)
        print(f"Network loaded from {filename}")

        return network

def train(network, train_loader, report_interval, num_epochs=1):
        # Enable network learning
        network.learning = True
        print({"Training Configuration": network})
        print('network learning: ', network.learning, flush=True)
        initial_weights = network.synapse_input_hidden.w.clone()

        for epoch in range(num_epochs):
            print(f"\nStarting Epoch {epoch+1}/{num_epochs}\n", flush=True)
            epoch_initial_weights = network.synapse_input_hidden.w.clone()

            #start training loop
            for idx, (image, label) in tqdm(enumerate(train_loader), desc='training', total=len(train_loader)):
                # Forward pass through the network
                outs = network.forward(image)
                # Report at intervals
                if idx % report_interval == 0:
                    max_spike_indices = torch.where(outs == outs.max())[0]
                    print(f"Sample {idx} - Label: {label} - Output Spikes: {outs} - Most spiked = {max_spike_indices}", flush=True)
                    print("Learned thetas: ", network.hidden_layer.theta, flush=True)
        
        print("Learned thetas: ", network.hidden_layer.theta, flush=True)
        last_weights = network.synapse_input_hidden.w.clone()

        print('weights changed: ', torch.all(torch.eq(initial_weights, last_weights)).item() == 0, flush=True)
        save_network(network, filename="not_assigned_network.pkl")


def assign_neuron_class_mapping(network, train_loader, report_interval):
    """
    Assigns each neuron in the hidden layer of the network to a class based on cumulative spike responses.
    
    The function processes the training dataset (using batch size 1 for compatibility) and accumulates
    the output spikes of each neuron for every class label. It then computes the probability distribution
    of spikes per neuron and assigns each neuron the class label for which it fired most consistently.
    
    Args:
        network (Network): The spiking neural network instance.
        train_loader (DataLoader): The training data loader.
        report_interval (int): Interval at which to report progress.
    
    Side Effects:
        Updates network.neuron_class_mapping with a dictionary mapping neuron indices to class labels.
        Saves the updated network using the save_network() function.
    """
    # Disable learning for evaluation and ensure no gradient tracking
    network.learning = False
    print("Network learning disabled:", network.learning, flush=True)
    print("Received theta values:", network.hidden_layer.theta, flush=True)

    # Save the initial T value for later resetting
    initial_T = network.T

    # Clone initial synapse weights for reference
    initial_weights = network.synapse_input_hidden.w.clone()

    # Get unique, sorted class labels from training data
    unique_labels = sorted(torch.unique(torch.tensor(train_loader.dataset.targets)).tolist())
    print("Unique labels:", unique_labels, flush=True)

    # Initialize cumulative spikes for each class (each is a tensor of zeros with length = hidden layer size)
    cumulative_spikes = {
        label: torch.zeros(network.hidden_size, device=network.device) for label in unique_labels
    }

    # Use no_grad to disable gradient computation during evaluation
    with torch.no_grad():
        # Accumulate spikes per neuron for each class across the training set
        for idx, (image, label) in tqdm(enumerate(train_loader), desc='Assigning neurons to classes', total=len(train_loader)):
            label = label.item()

            # First forward pass using the initial T value; result is a tensor
            outs = network.forward(image).flatten()  # stays on network.device
            max_val = torch.max(outs)
            max_indices = torch.where(outs == max_val)[0]

            # Check if the maximum spike is unique; if not, try resolving ties by increasing T
            if len(max_indices) != 1:
                for iteration in range(5):
                    network.T += 5
                    outs_new = network.forward(image).flatten()
                    max_val_new = torch.max(outs_new)
                    max_indices_new = torch.where(outs_new == max_val_new)[0]
                    if len(max_indices_new) == 1 or iteration == 4:
                        outs = outs_new  # Use the output from this iteration (unique or final iteration)
                        max_val = max_val_new
                        max_indices = max_indices_new
                        break

            # Accumulate only the spike output from the chosen run.
            cumulative_spikes[label] += outs

            # Reset T to its initial value after processing the sample
            network.T = initial_T

            if idx % report_interval == 0:
                max_spike_indices = torch.where(outs == torch.max(outs))[0]
                # Print outs by moving them to CPU for a clear display
                print(f"Sample {idx} - Label: {label} - Outputs: {outs.cpu().numpy()} - Most spiked indices: {max_spike_indices.cpu().numpy()}", flush=True)
                print("Theta values:", network.hidden_layer.theta, flush=True)

    # Compute per-label spike probabilities for each neuron
    probabilities = {}
    for label, spikes in cumulative_spikes.items():
        total_spikes = spikes.sum().item()
        if total_spikes > 0:  # Avoid division by zero
            probabilities[label] = (spikes / total_spikes).tolist()
        else:
            probabilities[label] = [0] * network.hidden_size

    # Build a mapping: for each neuron, compile a list of probabilities for each label (in sorted order)
    neuron_probs = {
        neuron_idx: [probabilities[label][neuron_idx] for label in unique_labels]
        for neuron_idx in range(network.hidden_size)
    }
    print("Neuron probabilities per class:", neuron_probs, flush=True)

    # Determine class assignment for each neuron based on maximum probability
    network.neuron_class_mapping = {}
    for neuron_idx, probs in neuron_probs.items():
        max_index, _ = max(enumerate(probs), key=lambda x: x[1])
        assigned_label = unique_labels[max_index]
        network.neuron_class_mapping[neuron_idx] = assigned_label

    # Report mapping for each neuron
    for neuron_idx, assigned_label in network.neuron_class_mapping.items():
        print(f"Neuron {neuron_idx} is assigned to class {assigned_label}", flush=True)

    # Check if weights have changed during this process
    last_weights = network.synapse_input_hidden.w.clone()
    weights_changed = not torch.all(torch.eq(initial_weights, last_weights)).item()
    print("Weights changed during assignment:", weights_changed, flush=True)

    print("Final neuron class mapping:", network.neuron_class_mapping, flush=True)
    save_network(network, filename="assigned_network.pkl")



def test(network, test_loader, report_interval):
    """
    Evaluate the network on the test dataset.

    This function performs inference on each sample in the test dataset, attempts to 
    resolve ambiguous outputs by increasing the simulation time T if necessary, and 
    computes the overall accuracy. It also plots and saves a confusion matrix.

    Args:
        network (Network): The spiking neural network instance.
        test_loader (DataLoader): DataLoader for the test dataset.
        report_interval (int): Interval at which progress is printed.

    Returns:
        float: The accuracy of the network on the test dataset.
    """
    # Set network to evaluation mode and store the initial simulation time
    network.learning = False
    initial_T = network.T
    print({"Testing Configuration": network})
    print("Network learning:", network.learning, flush=True)
    print("Hidden layer theta:", network.hidden_layer.theta, flush=True)

    # Retrieve unique class labels from test targets and move them to the device
    unique_labels = torch.tensor(test_loader.dataset.targets).unique().to(network.device)

    correct = 0
    total = 0
    y_true, y_pred = [], []

    # Use no_grad to save memory (even if gradients are not computed)
    with torch.no_grad():
        for idx, (image, label) in tqdm(enumerate(test_loader), desc='Testing', total=len(test_loader)):
            label_idx = label.item()
            predicted = None

            # First forward pass using the initial T value
            outs = network.forward(image).flatten().cpu().numpy()
            max_val = np.max(outs)
            max_indices = np.where(outs == max_val)[0]

            # Check if the maximum spike is unique
            if len(max_indices) == 1:
                predicted = network.neuron_class_mapping[max_indices[0]]
                unique = True
            else:
                unique = False
                # Try increasing T in fixed increments to resolve ties
                for iteration in range(5):
                    network.T += 5
                    outs = network.forward(image).flatten().cpu().numpy()
                    max_val = np.max(outs)
                    max_indices = np.where(outs == max_val)[0]
                    if len(max_indices) == 1:
                        predicted = network.neuron_class_mapping[max_indices[0]]
                        unique = True
                        break

            # Fallback if a unique maximum is not found
            if not unique and predicted is None:
                predicted = network.neuron_class_mapping[max_indices[0]]

            # Reset T to its initial value after processing the sample
            network.T = initial_T

            y_true.append(label_idx)
            y_pred.append(predicted)

            if predicted == label_idx:
                correct += 1
            total += 1

            if idx % report_interval == 0:
                print(f"Sample {idx} - Label: {label_idx} - Output Spike Counts: {outs}")
                print("Theta values:", network.hidden_layer.theta, flush=True)
                print(f"Predicted: {predicted}, Actual: {label_idx}")

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}", flush=True)

    # Generate and print the confusion matrix
    labels = unique_labels.tolist()
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:\n", conf_matrix, flush=True)

    # Save the confusion matrix plot
    os.makedirs('plots/train_test', exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('plots/train_test/confusion_matrix.png')
    print("Confusion Matrix saved to plots/train_test/confusion_matrix.png")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    return accuracy


def main():
    wandb.init(
            # set the wandb project where this run will be logged
            project="snn_training",

            # optional: give your run a short name
            name="corrected v_rest for 40",

            # optional: add a description of the run
            notes="Best accuracy so far with stdp corrected for 40k and 100 T. increased from 50 to 100 T and corrected v rest",

            # track hyperparameters and run metadata
            
            config={'num_train_data': 40000,
                    'num_test_data': 10000,
                    'report_interval': 1000,
                    'num_epochs': 1,
                    'T':100,
                    'hidden_size':500,
                    'tau':130,
                    'R':0.12,
                    'scale':10,
                    'dt':1,
                    'V_rest':0.99,
                    'theta':2,
                    'refrac':7,
                    'tau_theta':150,
                    'theta_increment': 30,
                    'A_plus': 0.00008,
                    'A_minus': 0.00008 * 1.052,
                    'tau_stdp':200,
                    'max_w':255,
                    'min_w':0,
                    'plot':False, 
                    'device':'cpu'}
        )
    
    config = wandb.config

    # Load MNIST dataset using PyTorch
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Limit dataset size for testing purposes
    train_dataset.data, train_dataset.targets = train_dataset.data[:config['num_train_data']], train_dataset.targets[:config['num_train_data']]
    # Limit dataset size for testing purposes
    test_dataset.data, test_dataset.targets = test_dataset.data[:config['num_test_data']], test_dataset.targets[:config['num_test_data']]

    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # Initialize DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    network = Network(T=config['T'],
                    hidden_size=config['hidden_size'],
                    tau=config['tau'],
                    R=config['R'],
                    scale=config['scale'],
                    dt=config['dt'],
                    V_rest=config['V_rest'],
                    theta=config['theta'],
                    refrac=config['refrac'],
                    tau_theta=config['tau_theta'],
                    theta_increment=config['theta_increment'],
                    A_plus=config['A_plus'],
                    A_minus=config['A_minus'],
                    tau_stdp=config['tau_stdp'],
                    max_w=config['max_w'],
                    min_w=config['min_w'],
                    plot=config['plot'],
                    device=config['device']
                )
    # Train network
    train(network, train_loader, config['report_interval'], config['num_epochs'])
    assign_neuron_class_mapping(network, train_loader, config['report_interval'])
    accuracy = test(network, test_loader, config['report_interval'])
    wandb.log({'accuracy': accuracy})
    wandb.finish()

if __name__ == '__main__':
    main()
