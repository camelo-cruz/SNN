import os
import pickle
import torch
import numpy as np
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def save_network(network, filename="network.pkl"):
    """
    Save the entire network object to a pickle file.

    Args:
        network: The network object to be saved.
        filename (str): The file path for saving the network.
    """
    with open(filename, 'wb') as f:
        pickle.dump(network, f)
    print(f"Network saved to {filename}")


def load_network(filename):
    """
    Load the entire network object from a pickle file.

    Args:
        filename (str): Path to the file from which the network will be loaded.
    
    Returns:
        The loaded network object.
    """
    with open(filename, 'rb') as f:
        network = pickle.load(f)
    print(f"Network loaded from {filename}")
    return network


def train(network, train_loader, report_interval, num_epochs=1):
    """
    Train the network using the provided training DataLoader.

    Args:
        network: The spiking neural network instance.
        train_loader (DataLoader): DataLoader for training data.
        report_interval (int): Interval (in samples) at which progress is reported.
        num_epochs (int): Number of epochs for training.
    """
    # Enable learning mode and report the current configuration.
    network.stdp = True
    network.homeostasis = True
    print({"Training Configuration": network})
    print("Network stdp:", network.stdp, flush=True)
    print("Network homeostasis:", network.homeostasis, flush=True)

    # Record initial synaptic weights.
    initial_weights = network.synapse_input_hidden.w.clone()

    for epoch in range(num_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{num_epochs}\n", flush=True)
        epoch_initial_weights = network.synapse_input_hidden.w.clone()

        # Training loop over batches.
        for idx, (image, label) in tqdm(enumerate(train_loader),
                                          desc='Training',
                                          total=len(train_loader)):
            # Forward pass through the network.
            outs = network.forward(image)

            # Report progress at specified intervals.
            if idx % report_interval == 0:
                max_spike_indices = torch.where(outs == outs.max())[0]
                print(f"Sample {idx} - Label: {label} - Output Spikes: {outs} - Most spiked: {max_spike_indices}",
                      flush=True)
                print("Learned thetas:", network.hidden_layer.theta, flush=True)

        print("Learned thetas after epoch:", network.hidden_layer.theta, flush=True)
        last_weights = network.synapse_input_hidden.w.clone()

        # Check and report if synaptic weights have changed.
        weights_changed = not torch.all(torch.eq(initial_weights, last_weights)).item()
        print("Weights changed:", weights_changed, flush=True)

    # Save network after training.
    save_network(network, filename="not_assigned_network.pkl")


def assign_neuron_class_mapping(network, train_loader, report_interval):
    """
    Assign each hidden neuron a class based on its cumulative spike response.

    The function runs through the training set (batch size 1) and accumulates spike counts per class.
    Then, it computes the per-neuron probability distributions and assigns each neuron the label 
    for which it fires most consistently.

    Args:
        network: The spiking neural network instance.
        train_loader (DataLoader): DataLoader containing training samples.
        report_interval (int): Interval at which progress is printed.
    """
    # Disable learning to ensure a fixed evaluation environment.
    network.stdp = False
    network.homeostasis = False
    print({"Training Configuration": network})
    print("Network stdp:", network.stdp, flush=True)
    print("Network homeostasis:", network.homeostasis, flush=True)

    # Save initial simulation time and synapse weights.
    initial_T = network.T
    initial_weights = network.synapse_input_hidden.w.clone()

    # Extract sorted unique class labels from the training dataset.
    unique_labels = sorted(torch.unique(torch.tensor(train_loader.dataset.targets)).tolist())
    print("Unique labels:", unique_labels, flush=True)

    # Dictionary for accumulating spikes per class.
    cumulative_spikes = {label: torch.zeros(network.hidden_size, device=network.device)
                         for label in unique_labels}

    with torch.no_grad():
        for idx, (image, label) in tqdm(enumerate(train_loader),
                                          desc='Assigning neurons to classes',
                                          total=len(train_loader)):
            label_val = label.item()

            # Run the forward pass with the current T.
            outs = network.forward(image).flatten()
            max_indices = torch.where(outs == outs.max())[0]

            # If there is ambiguity (tie), increase T in increments until a unique winner is found.
            if len(max_indices) != 1:
                for iteration in range(5):
                    network.T += 5
                    outs_new = network.forward(image).flatten()
                    max_indices_new = torch.where(outs_new == outs_new.max())[0]
                    if len(max_indices_new) == 1 or iteration == 4:
                        outs, max_indices = outs_new, max_indices_new
                        break

            # Accumulate spikes for the given class label.
            cumulative_spikes[label_val] += outs

            # Reset simulation time T.
            network.T = initial_T

            if idx % report_interval == 0:
                print(f"Sample {idx} - Label: {label_val} - Outputs: {outs.cpu().numpy()} - "
                      f"Most spiked indices: {max_indices.cpu().numpy()}", flush=True)
                print("Theta values:", network.hidden_layer.theta, flush=True)

    # Compute per-class spike probabilities for each neuron.
    probabilities = {}
    for label, spikes in cumulative_spikes.items():
        total_spikes = spikes.sum().item()
        if total_spikes > 0:
            probabilities[label] = (spikes / total_spikes).tolist()
        else:
            probabilities[label] = [0] * network.hidden_size

    # Build probability list per neuron.
    neuron_probs = {
        neuron_idx: [probabilities[label][neuron_idx] for label in unique_labels]
        for neuron_idx in range(network.hidden_size)
    }
    print("Neuron probabilities per class:", neuron_probs, flush=True)

    # Assign each neuron to the class with the highest probability.
    network.neuron_class_mapping = {}
    for neuron_idx, probs in neuron_probs.items():
        max_index, _ = max(enumerate(probs), key=lambda x: x[1])
        assigned_label = unique_labels[max_index]
        network.neuron_class_mapping[neuron_idx] = assigned_label

    # Report the final mapping.
    for neuron_idx, assigned_label in network.neuron_class_mapping.items():
        print(f"Neuron {neuron_idx} is assigned to class {assigned_label}", flush=True)

    # Check whether synaptic weights changed during the mapping process.
    last_weights = network.synapse_input_hidden.w.clone()
    weights_changed = not torch.all(torch.eq(initial_weights, last_weights)).item()
    print("Weights changed during assignment:", weights_changed, flush=True)
    print("Final neuron class mapping:", network.neuron_class_mapping, flush=True)

    # Save the updated network.
    save_network(network, filename="assigned_network.pkl")


def test(network, test_loader, report_interval):
    """
    Evaluate the network on test data and report classification accuracy,
    a confusion matrix, and a classification report.

    The function resolves ambiguous outputs by increasing T if necessary.

    Args:
        network: The spiking neural network instance.
        test_loader (DataLoader): DataLoader containing test samples.
        report_interval (int): Interval at which progress is printed.

    Returns:
        float: Classification accuracy.
    """
    network.stdp = False
    network.homeostasis = False
    print({"Training Configuration": network})
    print("Network stdp:", network.stdp, flush=True)
    print("Network homeostasis:", network.homeostasis, flush=True)
    initial_T = network.T
    print("Hidden layer theta:", network.hidden_layer.theta, flush=True)

    # Get the unique class labels from the test targets.
    unique_labels = torch.tensor(test_loader.dataset.targets).unique().to(network.device)

    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for idx, (image, label) in tqdm(enumerate(test_loader),
                                          desc='Testing',
                                          total=len(test_loader)):
            label_val = label.item()
            predicted = None

            # Forward pass with the initial T.
            outs = network.forward(image).flatten().cpu().numpy()
            max_val = np.max(outs)
            max_indices = np.where(outs == max_val)[0]

            # If only one neuron is most active, use its mapping.
            if len(max_indices) == 1:
                predicted = network.neuron_class_mapping[max_indices[0]]
                unique = True
            else:
                unique = False
                # Attempt to resolve ties by increasing T.
                for iteration in range(5):
                    network.T += 5
                    outs = network.forward(image).flatten().cpu().numpy()
                    max_val = np.max(outs)
                    max_indices = np.where(outs == max_val)[0]
                    if len(max_indices) == 1:
                        predicted = network.neuron_class_mapping[max_indices[0]]
                        unique = True
                        break

            # Fallback in case of continued ambiguity.
            if not unique and predicted is None:
                predicted = network.neuron_class_mapping[max_indices[0]]

            # Reset T to the initial value.
            network.T = initial_T

            y_true.append(label_val)
            y_pred.append(predicted)

            if predicted == label_val:
                correct += 1
            total += 1

            if idx % report_interval == 0:
                print(f"Sample {idx} - Label: {label_val} - Output Spike Counts: {outs}")
                print("Theta values:", network.hidden_layer.theta, flush=True)
                print(f"Predicted: {predicted}, Actual: {label_val}", flush=True)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}", flush=True)

    # Generate and display the confusion matrix.
    labels = unique_labels.tolist()
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
