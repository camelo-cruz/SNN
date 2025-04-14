import os
import torch
import matplotlib.pyplot as plt
from collections import deque
from Neuron import CurrentLIF  # Assumes CurrentLIF is defined in a module named "Neuron"


class Synapse:
    """
    A class representing a synapse connecting a presynaptic neuron to a postsynaptic neuron with STDP.
    
    The synapse maintains a weight matrix that is updated according to a spike-timing dependent plasticity (STDP)
    rule. It also implements a simple delay mechanism using a queue for spike transmission.
    """
    def __init__(self, pre_neuron, post_neuron, w=None, max_w=255, min_w=0, A_plus=0.005, A_minus=0.005, tau_stdp=40, device='cpu', record_history=False):
        """
        Initialize the synapse with specified STDP parameters.
        
        Args:
            pre_neuron (CurrentLIF): The presynaptic neuron.
            post_neuron (CurrentLIF): The postsynaptic neuron.
            max_w (float): Maximum allowed synaptic weight.
            min_w (float): Minimum allowed synaptic weight.
            A_plus (float): LTP learning rate.
            A_minus (float): LTD learning rate.
            tau_stdp (float): Time constant for trace decay in STDP.
            device (str or torch.device): Device to use for tensor computations.
            record_history (bool): Whether to record the synaptic weight history.
        """
        print(f"Synapse device selected: {device}")
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.max_w = max_w
        self.min_w = min_w
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_stdp = tau_stdp
        self.device = device
        self.record_history = record_history


        self.w = w
        print(f"Initialized synaptic weights: {self.w}", flush=True)

        # Record initial weight if history recording is enabled.
        self.w_history = [self.w.clone()] if record_history else None

        # Initialize STDP traces for pre- and postsynaptic neurons.
        self.pre_trace = torch.zeros(pre_neuron.batch_size, device=device)
        self.post_trace = torch.zeros(post_neuron.batch_size, device=device)

        # Spike queue for implementing a simple delay in spike transmission.
        self.spike_queue = deque(maxlen=2)

    def transmit(self):
        """
        Transmit spikes from the presynaptic neuron to the postsynaptic neuron using a delay queue.
        
        The synaptic output is computed as the dot product between the presynaptic spike vector and the weight
        matrix. This output is queued and transmitted with a delay.
        """
        synaptic_output = self.pre_neuron.spiked.float() @ self.w
        self.spike_queue.append(synaptic_output)

        if len(self.spike_queue) > 1:
            delayed_output = self.spike_queue.popleft()
            self.post_neuron.add_input(delayed_output)

    def update_stdp(self):
        """
        Update the synaptic weights using spike-timing dependent plasticity (STDP).
        
        This method performs the following steps:
            1. Decay the existing pre- and postsynaptic traces.
            2. Update the traces with the new spike events.
            3. Compute the weight changes:
               - LTP: Increase weights proportional to the presynaptic trace when a postsynaptic spike occurs.
               - LTD: Decrease weights proportional to the postsynaptic trace when a presynaptic spike occurs.
            4. Clamp the weights to remain within specified bounds.
            5. Record the weight history if enabled.
        """
        # Compute decay factor for the STDP traces.
        decay_factor = torch.exp(torch.tensor(-1.0 / self.tau_stdp, device=self.device))
        self.pre_trace *= decay_factor
        self.post_trace *= decay_factor

        # Update traces using current spikes.
        pre_spiked = self.pre_neuron.spiked.float()   # (N_pre,)
        post_spiked = self.post_neuron.spiked.float()   # (N_post,)
        self.pre_trace += pre_spiked
        self.post_trace += post_spiked

        # Long-Term Potentiation (LTP): Increase weight when postsynaptic spike occurs.
        delta_w_LTP = self.A_plus * self.pre_trace.unsqueeze(1) * post_spiked.unsqueeze(0) * (self.max_w - self.w)

        # Long-Term Depression (LTD): Decrease weight when presynaptic spike occurs.
        delta_w_LTD = self.A_minus * pre_spiked.unsqueeze(1) * self.post_trace.unsqueeze(0) * (self.w - self.min_w)

        # Update synaptic weights.
        self.w += (delta_w_LTP - delta_w_LTD)
        self.w = torch.clamp(self.w, self.min_w, self.max_w)

        # Record weight history if enabled.
        if self.record_history:
            self.w_history.append(self.w.clone())

    def plot(self, title='Weight and Spike Evolution', plot_dir='plots/synapse_simulation'):
        """
        Plot the evolution of the synaptic weights along with the spike times and post-neuron state.
        
        For each pair of connected neurons, it creates a figure with three subplots:
            1. Presynaptic spike times.
            2. Postsynaptic membrane potential and dynamic threshold.
            3. Synaptic weight evolution.
        The plots are saved to the specified directory.
        
        Args:
            title (str): Title prefix for the saved plot files.
            plot_dir (str): Directory in which to save the plots.
        """
        if self.w_history is None:
            print("No weight history to plot.")
            return

        os.makedirs(plot_dir, exist_ok=True)

        # Iterate over all synapse pairs between pre- and postsynaptic neurons.
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                # Extract weight history for the connection (i -> j).
                weights = [w[i, j].item() for w in self.w_history]

                # Identify time indices when the presynaptic neuron i spiked.
                pre_spike_times = [idx for idx, spk in enumerate(self.pre_neuron.spike_history) if spk[i]]

                # Create a figure with three subplots.
                fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

                # 1. Pre-synaptic spike times.
                axs[0].scatter(pre_spike_times, [1] * len(pre_spike_times), label=f'Pre Spike Neuron {i}', color='blue', marker='o')
                axs[0].set_ylabel('Spikes')
                axs[0].set_title(f'{title} - Pre Spike Neuron {i} → Post Neuron {j}')
                axs[0].legend()
                axs[0].set_ylim(0, 2)

                # 2. Postsynaptic membrane potential and threshold.
                if hasattr(self.post_neuron, 'V_history') and hasattr(self.post_neuron, 'theta_history'):
                    post_history = [state[j].item() for state in self.post_neuron.V_history]
                    theta_history = [thr[j].item() for thr in self.post_neuron.theta_history]

                    axs[1].plot(post_history, label=f'Post V (Neuron {j})', color='red')
                    axs[1].plot(theta_history, label=f'Threshold (Neuron {j})', color='purple', linestyle='dashed')
                    axs[1].set_ylabel('Potential / Threshold')
                    axs[1].set_title(f'{title} - Post Neuron {j} State')
                    axs[1].legend(loc='upper left')

                # 3. Synaptic weight evolution.
                axs[2].plot(weights, label='Synaptic Weight', color='green')
                axs[2].set_xlabel('Time Step')
                axs[2].set_ylabel('Weight')
                axs[2].set_title(f'{title} - Weight (Pre {i} → Post {j})')
                axs[2].legend()

                plt.tight_layout()
                fig_path = os.path.join(plot_dir, f'weight_and_spikes_pre_{i}_post_{j}.png')
                plt.savefig(fig_path)
                plt.close()

        print('Synapse weight, spike evolution, and post-neuron state plotted.')


def simulation():
    """
    Run a simulation for a pair of neurons connected by a synapse with STDP.
    
    The simulation runs for a number of time steps (epochs). At each epoch, a spike input is generated for
    the presynaptic neuron, which is then updated. The synapse transmits the spike to the postsynaptic neuron,
    and STDP updates are applied to the synaptic weight. Finally, the evolution of the synapse is plotted.
    """
    # Initialize pre- and postsynaptic neurons.
    pre_neuron = CurrentLIF(batch_size=1, scale=5, record_history=True, device='cpu')
    post_neuron = CurrentLIF(batch_size=1, scale=1, record_history=True, device='cpu')

    # Create the synapse with specified STDP parameters.
    synapse = Synapse(
        pre_neuron=pre_neuron,
        post_neuron=post_neuron,
        max_w=5.0,
        min_w=1.0,
        A_plus=0.005,
        A_minus=0.006,
        tau_stdp=20,
        record_history=True,
        device='cpu'
    )

    num_epochs = 100
    # Generate a random spike train (Bernoulli process) for the presynaptic neuron.
    spike_train = torch.bernoulli(torch.full((num_epochs,), 0.5)).to('cpu')

    # Run the simulation over the number of epochs.
    for epoch in range(num_epochs):
        # Generate input from the spike train.
        current_input = spike_train[epoch]
        pre_neuron.add_input(current_input)
        pre_neuron.update()
        synapse.transmit()
        post_neuron.update()
        synapse.update_stdp()

    # Generate and save the plot(s).
    synapse.plot('Synapse Simulation')


if __name__ == '__main__':
    simulation()
