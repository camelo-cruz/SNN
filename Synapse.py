import os
import torch
from Neuron import CurrentLIF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class Synapse:
    def __init__(self, pre_neuron, post_neuron, max_w, min_w, A_plus, A_minus, tau_stdp, device, record_history=False):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.max_w = max_w
        self.min_w = min_w
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_stdp = tau_stdp
        self.device = device
        self.record_history = record_history

        mean_w = (max_w + min_w) / 2
        self.w = torch.normal(
            mean=mean_w, 
            std=0.1, 
            size=(pre_neuron.batch_size, post_neuron.batch_size), 
            device=device
        )
        self.w = torch.clamp(self.w, min_w, max_w)

        if self.record_history:
            self.w_history = [self.w.clone()]

        self.pre_trace = torch.zeros(pre_neuron.batch_size, device=device)
        self.post_trace = torch.zeros(post_neuron.batch_size, device=device)

        self.spike_queue = deque(maxlen=2)

    def transmit(self):
        self.spike_queue.append(self.pre_neuron.spiked.float() @ self.w)

        if len(self.spike_queue) > 1:
            delayed_spikes = self.spike_queue.popleft()
            self.post_neuron.add_input(delayed_spikes)

    def update_stdp(self):
        # 1) Decay the old traces
        decay_factor = torch.exp(torch.tensor(-1.0 / self.tau_stdp, device=self.device))
        self.pre_trace *= decay_factor
        self.post_trace *= decay_factor

        # 2) Add current spikes to the traces (so that weight updates see the *new* increments)
        pre_spiked = self.pre_neuron.spiked.float()   # shape: (N_pre,)
        post_spiked = self.post_neuron.spiked.float() # shape: (N_post,)
        self.pre_trace += pre_spiked
        self.post_trace += post_spiked

        # 3) Compute weight changes using the updated traces

        # LTP: for each postsyn neuron j that spikes, w[i,j] += A_plus * x_i
        # In vector form: delta_w_pot = A_plus * outer(pre_trace, post_spiked)
        # the weight is increased at the moment of postsynaptic firing by an amount that depends on the value of the trace x left by the presynaptic spike.
        delta_w_LTP = self.A_plus * self.pre_trace.unsqueeze(1) * post_spiked.unsqueeze(0)

        # LTD: for each presyn neuron i that spikes, w[i,j] -= A_minus * y_j
        # In vector form: delta_w_dep = A_minus * outer(pre_spiked, post_trace)
        # the weight is depressed at the moment of presynaptic spikes by an amount proportional to the trace y left by previous postsynaptic spikes.
        delta_w_LTD = self.A_minus * pre_spiked.unsqueeze(1) * self.post_trace.unsqueeze(0)

        # Apply the updates
        self.w += (delta_w_LTP - delta_w_LTD)

        # 4) Optional weight clamping
        self.w = torch.clamp(self.w, self.min_w, self.max_w)

        # Record weight history if enabled
        if self.record_history:
            self.w_history.append(self.w.clone())

    def plot(self, title='Weight and Spike Evolution', plot_dir='plots/synapse_simulation'):
        if not self.w_history:
            print("No weights to plot.")
            return

        os.makedirs(plot_dir, exist_ok=True)

        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                weights = [w[i, j].item() for w in self.w_history]

                pre_spike_times = [idx for idx, spk in enumerate(self.pre_neuron.spike_history) if spk[i]]

                fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)  # Keep 3 subplots

                # Plot pre-synaptic spike times
                axs[0].scatter(pre_spike_times, [1] * len(pre_spike_times), label=f'Pre Spike Neuron {i}', color='blue', marker='o')
                axs[0].set_ylabel('Spikes')
                axs[0].set_title(f'{title} - Pre Spike Neuron {i} -> Post Neuron {j}')
                axs[0].legend()
                axs[0].set_ylim(0, 2)

                # Plot post-neuron membrane potential and threshold together
                if hasattr(self.post_neuron, 'V_history') and hasattr(self.post_neuron, 'theta_history'):
                    post_history = [post[j].item() for post in self.post_neuron.V_history]
                    theta_history = [theta[j].item() for theta in self.post_neuron.theta_history]

                    axs[1].plot(post_history, label=f'Post Membrane Potential Neuron {j}', color='red')
                    axs[1].plot(theta_history, label=f'Threshold (Theta) Neuron {j}', color='purple', linestyle='dashed')

                    axs[1].set_ylabel('Potential / Threshold')
                    axs[1].set_title(f'{title} - Membrane Potential & Threshold for Post Neuron {j}')
                    axs[1].legend(loc='upper left')

                # Plot synaptic weight evolution
                axs[2].plot(weights, label='Synaptic Weight', color='green')
                axs[2].set_xlabel('Time Step')
                axs[2].set_ylabel('Weight')
                axs[2].set_title(f'{title} Weight Pre {i} -> Post {j}')
                axs[2].legend()

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'weight_and_spikes_pre_{i}_post_{j}.png'))
                plt.close()

        print('Synapse weight, spike evolution, and post-neuron potential with threshold plotted')



def simulation():
    pre_neuron = CurrentLIF(batch_size=1, scale=5, record_history=True, device='cpu')
    post_neuron = CurrentLIF(batch_size=1, scale=1, record_history=True, device='cpu')

    synapse = Synapse(pre_neuron=pre_neuron, post_neuron=post_neuron, max_w=5.0, min_w=1.0, A_plus=0.005, A_minus=0.006, tau_stdp=20, record_history=True, device='cpu')

    num_epochs = 100
    spike_train = torch.bernoulli(torch.full((num_epochs,), 0.5)).to('cpu')

    for epoch in range(num_epochs):
        I = spike_train[epoch]
        pre_neuron.add_input(I)
        pre_neuron.update()
        synapse.transmit()
        post_neuron.update()
        synapse.update_stdp()

    synapse.plot('Synapse Simulation')


if __name__ == '__main__':
    simulation()
