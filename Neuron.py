import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Neuron:
    def __init__(self, tau=100, R=1, scale=1, dt=1, record_history=False, homeostasis=False, device='cuda', batch_size=1):
        self.tau = tau
        self.R = R
        self.scale = scale
        self.dt = dt
        self.record_history = record_history
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')  # Use CUDA if available
        print(f"Neuron device selected: {self.device}")
        self.batch_size = batch_size
        self.homeostasis = homeostasis
       
    def reset(self):
        """Reset the neuron after a spike or series of spikes"""
        raise NotImplementedError("The 'reset' method should be implemented in subclasses.")

    def add_input(self, I):
        raise NotImplementedError("The 'add_input' method should be implemented in subclasses.")

    def update(self):
        raise NotImplementedError("The 'update' method should be implemented in subclasses.")

    def plot(self, title):
        if self.V_history is None or self.theta_history is None:
            print("No history recorded.")
            return

        V_history_tensor = torch.stack(self.V_history, dim=0)  # Shape: [time_steps, batch_size]
        theta_history_tensor = torch.stack(self.theta_history, dim=0)  # Shape: [time_steps, batch_size]

        num_neurons = V_history_tensor.shape[1]

        for neuron_index in range(num_neurons):
            plt.figure()
            plt.plot(V_history_tensor[:, neuron_index].cpu().numpy(), label="Membrane Potential")
            plt.plot(theta_history_tensor[:, neuron_index].cpu().numpy(), color='red', linestyle='--', label="Threshold")

            plt.title(f'Membrane Potential and Dynamic Threshold Over Time - Neuron {neuron_index}')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')
            plt.legend()

            save_dir = os.path.join('plots', 'neuron_simulation')
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{title}_neuron_{neuron_index}.png")
            plt.savefig(save_path)
            print(f'Saved plot for neuron {neuron_index}')
            plt.close()


class CurrentLIF(Neuron):
    def __init__(self, V_rest=0, theta=0.1, scale=1, refrac=5, tau_theta=30, theta_increment=0.1, batch_size=1, homeostasis=False, **kwargs):
        super().__init__(batch_size=batch_size, homeostasis=homeostasis, scale=scale, **kwargs)
        self.V_rest = V_rest
        self.V_reset = V_rest
        self.V = torch.full((self.batch_size,), V_rest, device=self.device, dtype=torch.float32)  # Membrane potential (mV)
        self.theta = torch.full((self.batch_size,), theta, device=self.device, dtype=torch.float32)  # Dynamic threshold
        self.refractory_period = refrac  # Refractory period (ms)
        self.refractory_timer = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)  # Refractory timer
        # Dynamic threshold parameters
        self.theta_init = theta
        self.tau_theta = tau_theta
        self.theta_increment = theta_increment
        self.spiked = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.I_syn = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)
        
        if self.record_history:
            self.V_history = []
            self.theta_history = []
            self.spike_history = []
        else:
            self.V_history = None
            self.theta_history = None
            self.spike_history = None

        # Precompute constants
        self.dt_over_tau = self.dt / self.tau
        self.dt_over_tau_theta = self.dt / self.tau_theta
        self.exp_decay_theta = torch.exp(torch.tensor(-self.dt_over_tau_theta, device=self.device))

    def add_input(self, I):
        """Add synaptic input current (nA) to the neuron."""
        self.I_syn += I.to(self.device) * self.scale  # Ensure I is on the correct device

    def update(self):
        """Update membrane potential using the LIF model."""
        # Identify neurons in the refractory period
        self.spiked.fill_(False)
        refractory_mask = self.refractory_timer > 0
        self.refractory_timer[refractory_mask] -= self.dt

        # Set the potential of refractory neurons to resting potential
        self.V[refractory_mask] = self.V_rest

        # Neurons not in the refractory period
        active_mask = ~refractory_mask

        # Update membrane potential for active neurons
        if active_mask.any():
            V_diff = self.V_rest - self.V[active_mask] + self.I_syn[active_mask] * self.R
            dV = self.dt_over_tau * V_diff
            self.V[active_mask] += dV
        
        # Record voltage and threshold history
        if self.record_history:
            self.V_history.append(self.V.clone())
            self.theta_history.append(self.theta.clone())
        
        # Check for neurons that have reached the threshold
        spike_mask = self.V >= self.theta
        
        #lateral inhibitioni
        # Winner-Takes-All Mechanismus
        if spike_mask.any():
            spike_indices = torch.where(spike_mask)[0]
            #winner with highest V
            winner_idx = spike_indices[torch.argmax(self.V[spike_indices])]

            self.spiked = torch.zeros_like(self.spiked)
            self.spiked[winner_idx] = True

            # Reset membrane potential and set refractory period for winnners
            self.V[winner_idx] = self.V_reset
            self.refractory_timer[winner_idx] = self.refractory_period

            # Reset membrane potential for non-winners
            self.V[~self.spiked] = self.V_reset
            self.refractory_timer[~self.spiked] = 1
        
        if self.record_history:
            self.spike_history.append(self.spiked.clone())

        if self.homeostasis:
            self.theta[self.spiked] += self.theta_increment
            # Exponential decay of thresholds
            self.theta = self.theta_init + (self.theta - self.theta_init) * self.exp_decay_theta
        
        # Reset synaptic input current
        self.I_syn.zero_()


    def reset(self):
        """Reset the neuron after a spike or series of spikes."""
        self.I_syn.zero_()
        self.V.fill_(self.V_reset)
        self.refractory_timer.zero_()


def simulation():
    # Initialisiere prä- und postsynaptische Neuronen
    pre_neuron = CurrentLIF(batch_size=1, scale=1, homeostasis=True, record_history=True)

    # Anzahl der Simulationsepochen (Zeitschritte)
    num_epochs = 10
    
    # Wahrscheinlichkeit, mit der ein Neuron spikt
    spike_train1 = torch.tensor([1,0,1,1,0,0,0,0,0,0], dtype=torch.float32)
    spike_train2 = torch.tensor([1,0,1,1,0,0,0,0,0,0], dtype=torch.float32)
    spike_train3 = torch.tensor([1,0,1,1,0,0,0,0,0,0], dtype=torch.float32)
    spike_train4 = torch.tensor([1,0,1,1,0,0,0,0,0,0], dtype=torch.float32)
    spike_train5 = torch.tensor([1,0,1,1,0,0,0,0,0,0], dtype=torch.float32)
    
    for epoch in range(num_epochs):
        pre_neuron.add_input(spike_train1[epoch])
        pre_neuron.add_input(spike_train2[epoch])
        pre_neuron.add_input(spike_train3[epoch])
        pre_neuron.add_input(spike_train4[epoch])
        pre_neuron.add_input(spike_train4[epoch])
        
        pre_neuron.update()
        print(pre_neuron.spiked)
    
    # Plot der Neuronenhistorie
    pre_neuron.plot('Pre Neuron')

if __name__ == '__main__':
    simulation()