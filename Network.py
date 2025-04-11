from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from Neuron import CurrentLIF 
from Receptive_Field import ReceptiveField
from Synapse import Synapse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Network:
    """
    A spiking neural network composed of an input layer (modeled as a ReceptiveField),
    a hidden layer (modeled as a group of CurrentLIF neurons), and synapses connecting
    the two layers that update via spike-timing dependent plasticity (STDP).
    """
    def __init__(self,
                 T=10,
                 input_size=28 * 28,
                 hidden_size=100,
                 tau=100,
                 R=1,
                 scale=1,
                 dt=1,
                 V_rest=0,
                 theta=0.5,
                 refractory_period=5,
                 tau_theta=30,
                 theta_increment=1,
                 # Synapse parameters
                 max_w=10,
                 min_w=0,
                 A_plus=0.005,
                 A_minus=0.005,
                 tau_stdp=20,
                 record_history=False,
                 plot=False,
                 device='cpu'):
        """
        Initialize the network with given parameters for neurons and synapses.

        Args:
            T (int): Number of time steps per forward pass.
            input_size (int): Size of the input (number of pixels).
            hidden_size (int): Number of hidden neurons.
            tau (int): Membrane time constant.
            R (int/float): Membrane resistance.
            scale (int/float): Scale factor for input current.
            dt (int/float): Simulation time step.
            V_rest (float): Resting membrane potential.
            theta (float): Initial dynamic threshold.
            refrac (int): Refractory period duration.
            tau_theta (int): Time constant for threshold decay.
            theta_increment (int/float): Increment of threshold on spike.
            max_w (float): Maximum synaptic weight.
            min_w (float): Minimum synaptic weight.
            A_plus (float): LTP learning rate.
            A_minus (float): LTD learning rate.
            tau_stdp (float): STDP time constant for trace decay.
            plot (bool): Whether to record and plot activity/history.
            device (str): The device to use (e.g., 'cpu' or 'cuda').
        """
        # General network parameters
        print(f"Network device selected: {device}")
        self.T = T
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.R = R
        self.scale = scale
        self.dt = dt
        self.V_rest = V_rest
        self.theta = theta
        self.refractory_period = refractory_period
        self.tau_theta = tau_theta
        self.theta_increment = theta_increment

        # Synapse parameters
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_stdp = tau_stdp
        self.max_w = max_w
        self.min_w = min_w

        # Recording and device settings
        self.record_history = record_history
        self.plot = plot
        self.device = device

        # These will be set after training
        self.class_neuron_mapping = None
        self.neuron_class_mapping = None

        # Initialize neurons and synapses
        self.create_neurons()
        self.create_synapses()

        # Internal flag controlling learning updates
        self._stdp = False
        self._homeostasis = False

    @property
    def stdp(self):
        """Flag indicating whether learning (STDP updates) is active."""
        return self._stdp
    
    @property
    def homeostasis(self):
        """Flag indicating whether learning (STDP updates) is active."""
        return self._homeostasis

    @stdp.setter
    def stdp(self, value):
        self._stdp = value
        print(f"stdp activated: {value}")
    
    @homeostasis.setter
    def homeostasis(self, value):
        self._homeostasis = value
        self.hidden_layer.homeostasis = value
        print(f"Homeostasis activated: {self.hidden_layer.homeostasis}")

    def __repr__(self):
        return (
            f"Network(T={self.T}, hidden_size={self.hidden_size}, tau={self.tau}, R={self.R}, "
            f"scale={self.scale}, dt={self.dt}, V_rest={self.V_rest}, theta={self.theta}, "
            f"refrac={self.refractory_period}, tau_theta={self.tau_theta}, theta_increment={self.theta_increment}, "
            f"A_plus={self.A_plus}, A_minus={self.A_minus}, tau_stdp={self.tau_stdp}, "
            f"max_w={self.max_w}, min_w={self.min_w}, record_history={self.record_history}, device='{self.device}')"
            f"stdp={self.stdp}, homeostasis={self.homeostasis}"
        )

    def clear_neurons(self):
        """
        Reset the internal state of all neurons and clear recorded histories if any.
        """
        self.hidden_layer.reset()
        if self.record_history:
            self.hidden_layer.V_history.clear()
            self.hidden_layer.theta_history.clear()
            self.synapse_input_hidden.w_history.clear()

    def create_neurons(self):
        """
        Initialize the input and hidden layers.
        The input layer is modeled as a ReceptiveField, and the hidden layer consists 
        of CurrentLIF neurons.
        """
        self.input_layer = ReceptiveField(
            input_size=self.input_size,
            record_history=self.record_history,
            device=self.device  # Use the provided device
        )

        self.hidden_layer = CurrentLIF(
            V_rest=self.V_rest,
            scale=self.scale,
            theta=self.theta,
            refractory_period=self.refractory_period,
            tau_theta=self.tau_theta,
            theta_increment=self.theta_increment,
            record_history=self.record_history,
            batch_size=self.hidden_size,
            device=self.device
        )

    def create_synapses(self):
        """
        Create synapses connecting the input layer to the hidden layer.
        """
        self.synapse_input_hidden = Synapse(
            pre_neuron=self.input_layer,
            post_neuron=self.hidden_layer,
            A_plus=self.A_plus,
            A_minus=self.A_minus,
            tau_stdp=self.tau_stdp,
            max_w=self.max_w,
            min_w=self.min_w,
            record_history=self.record_history,
            device=self.device
        )

    def forward(self, image):
        """
        Process an input image and propagate it through the network.

        The input is first encoded and processed by the input layer. Then, the hidden layer
        receives transmitted spikes via the synapses. The STDP updates are applied if learning is enabled.
        
        Args:
            image (torch.Tensor): The input image tensor.
        
        Returns:
            torch.Tensor: The cumulative spike count for the hidden neurons.
        """
        # Encode input image into potential over T time steps.
        self.input_layer.compute_potential(image, encoding='isi', time_steps=self.T)
        count_spikes = torch.zeros(self.hidden_size, device=self.device)

        # Process T time steps.
        for t in range(self.T):
            self.input_layer.fire(t)  # Generate spikes in the input layer.
            self.synapse_input_hidden.transmit()  # Transmit spikes to hidden neurons.
            self.hidden_layer.update()            # Update hidden neuron states.
            count_spikes += self.hidden_layer.spiked  # Accumulate spikes.

            # Apply STDP updates if learning is enabled.
            if self.stdp:
                self.synapse_input_hidden.update_stdp()

        if self.plot:
            self.plot_activity()

        self.clear_neurons()
        return count_spikes

    def plot_activity(self):
        """
        Plot the evolution of the synaptic weights and neuronal spike activity.
        """
        self.synapse_input_hidden.plot('input-hidden synapse')


def network_simulation():
    """
    Run a simple network simulation with a two-pixel input and two hidden neurons.

    An input image is provided to the network and spiking activity is propagated.
    After processing, the spike counts of the hidden neurons and their final thresholds are printed.
    """
    T = 10
    # For simplicity, we use a small tensor for the image.
    image = torch.tensor([[255, 255]], dtype=torch.float32)

    # Initialize network with a small input and hidden layer for demonstration.
    net = Network(T=T, input_size=2, hidden_size=2, scale=1, plot=True, device='cpu')
    net.homeostasis = True
    net.stdp = True

    outputs = net.forward(image)
    print("Output Spike Counts:", outputs)
    print("Final Theta Values:", net.hidden_layer.theta)


if __name__ == '__main__':
    network_simulation()