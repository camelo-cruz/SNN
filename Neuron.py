import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CurrentLIF:
    def __init__(self,
                 V_rest=0,
                 theta=0.1,
                 scale=1,
                 refractory_period=5,
                 tau_theta=30,
                 theta_increment=0.1,
                 batch_size=1,
                 homeostasis=False,
                 tau=20.0,
                 R=1.0,
                 dt=1.0,
                 record_history=False,
                 device='cpu'):
        """
        Initialize a current-based leaky integrate-and-fire neuron with a dynamic threshold.

        Args:
            V_rest (float): Resting membrane potential.
            theta (float): Initial threshold potential.
            scale (float): Scale factor for input current.
            refractory_period (float): Refractory period duration.
            tau_theta (float): Time constant for threshold decay.
            theta_increment (float): Increment value for threshold on spiking.
            batch_size (int): Number of neurons processed simultaneously.
            homeostasis (bool): Flag to enable homeostatic threshold adjustment.
            tau (float): Membrane time constant.
            R (float): Membrane resistance.
            dt (float): Simulation time step.
            record_history (bool): Whether to record membrane potential and threshold over time.
            device (str or torch.device, optional): Specify a device; if None, auto-select based on CUDA availability.
        """
        print(f"Neuron device selected: {device}")
        self.device = device

        # Save simulation and neuron parameters.
        self.batch_size = batch_size
        self.homeostasis = homeostasis
        self.V_rest = V_rest
        self.V_reset = V_rest
        self.tau = tau
        self.R = R
        self.dt = dt
        self.scale = scale
        self.refractory_period = refractory_period
        self.tau_theta = tau_theta
        self.theta_increment = theta_increment
        self.record_history = record_history

        # Precompute constants.
        self.dt_over_tau = self.dt / self.tau
        self.dt_over_tau_theta = self.dt / self.tau_theta

        # Set initial state for membrane potential and threshold.
        self.V = torch.full((self.batch_size,), V_rest, device=self.device, dtype=torch.float32)
        self.theta = torch.full((self.batch_size,), theta, device=self.device, dtype=torch.float32)
        self.theta_init = theta
        self.exp_decay_theta = torch.exp(torch.tensor(-self.dt_over_tau_theta, device=self.device))

        # Initialize timers and input holders.
        self.refractory_timer = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)
        self.spiked = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.I_syn = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)

        # Set up recording lists if enabled.
        if self.record_history:
            self.V_history = []
            self.theta_history = []
            self.spike_history = []
        else:
            self.V_history = None
            self.theta_history = None
            self.spike_history = None

    def add_input(self, I):
        """
        Add synaptic input current (nA) to the neuron.

        Args:
            I (torch.Tensor): Input current. It will be moved to the correct device.
        """
        self.I_syn += I.to(self.device) * self.scale

    def update(self):
        """
        Update the membrane potential using the leaky integrate-and-fire (LIF)
        dynamics with a dynamic threshold. Also manages refractory periods and
        implements a winner-takes-all mechanism for spiking.
        """
        # Reset spiked state for this update.
        self.spiked.fill_(False)

        # Update refractory timers and set membrane potential for neurons in refractory period.
        refractory_mask = self.refractory_timer > 0
        self.refractory_timer[refractory_mask] -= self.dt
        self.V[refractory_mask] = self.V_rest

        # Compute update only for active (non-refractory) neurons.
        active_mask = ~refractory_mask
        if active_mask.any():
            V_diff = self.V_rest - self.V[active_mask] + self.I_syn[active_mask] * self.R
            dV = self.dt_over_tau * V_diff
            self.V[active_mask] += dV

        # Record the state if history recording is enabled.
        if self.record_history:
            self.V_history.append(self.V.clone())
            self.theta_history.append(self.theta.clone())

        # Identify neurons reaching the dynamic threshold.
        spike_mask = self.V >= self.theta

        # Winner-takes-all mechanism: Only the neuron with the highest potential spikes.
        if spike_mask.any():
            spike_indices = torch.where(spike_mask)[0]
            winner_idx = spike_indices[torch.argmax(self.V[spike_indices])]

            self.spiked.zero_()
            self.spiked[winner_idx] = True

            # Reset membrane potentials and set refractory periods accordingly.
            self.V[winner_idx] = self.V_reset
            self.refractory_timer[winner_idx] = self.refractory_period
            self.V[~self.spiked] = self.V_reset
            self.refractory_timer[~self.spiked] = 1  # a short refractory period for non-winners

        if self.record_history:
            self.spike_history.append(self.spiked.clone())

        # Update dynamic threshold via homeostasis.
        if self.homeostasis:
            self.theta[self.spiked] += self.theta_increment
            self.theta = self.theta_init + (self.theta - self.theta_init) * self.exp_decay_theta

        # Clear the synaptic input.
        self.I_syn.zero_()

    def reset(self):
        """Reset the neuron's membrane potential, synaptic input, and refractory timer."""
        self.I_syn.zero_()
        self.V.fill_(self.V_reset)
        self.refractory_timer.zero_()

    def plot(self, title):
        """
        Generate and save plots for the membrane potential and dynamic threshold history.
        
        Args:
            title (str): Title used in the saved file names.
        """
        if self.V_history is None or self.theta_history is None:
            print("No history recorded.")
            return

        V_history_tensor = torch.stack(self.V_history, dim=0)   # shape: [time_steps, batch_size]
        theta_history_tensor = torch.stack(self.theta_history, dim=0)   # shape: [time_steps, batch_size]
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


def simulation():
    """
    Run a simple simulation over a number of epochs. A single neuron's input is
    updated with several spike trains, and the neuron's state is updated accordingly.
    """
    # Create a neuron with history recording and homeostatic threshold adjustment enabled.
    pre_neuron = CurrentLIF(batch_size=1, scale=1, homeostasis=True, record_history=True)

    num_epochs = 10

    # Create a list of spike trains.
    spike_trains = [
        torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        for _ in range(5)
    ]

    # Run simulation over epochs.
    for epoch in range(num_epochs):
        # Add input from each spike train for the current epoch.
        for train in spike_trains:
            pre_neuron.add_input(train[epoch])
        pre_neuron.update()
        print(f"Epoch {epoch}: Spiked state {pre_neuron.spiked.cpu().numpy()}")

    # Generate and save plots.
    pre_neuron.plot('Pre_Neuron')


if __name__ == '__main__':
    simulation()
