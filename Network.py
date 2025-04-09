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
    def __init__(self,
                 T=10,
                 input_size=28*28,
                 hidden_size=100,
                 tau=100,
                 R=1,
                 scale=1, 
                 dt=1,
                 V_rest=0, 
                 theta=0.5,
                 refrac=5, 
                 tau_theta=30, 
                 theta_increment=1,
                 #synapse parameters
                 max_w=10, 
                 min_w=0,
                 A_plus=0.005,
                 A_minus=0.005, 
                 tau_stdp=20,
                 plot=False,
                 device='cpu'):
        
        self.T = T 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.R = R
        self.scale = scale 
        self.dt = dt
        self.V_rest = V_rest
        self.theta = theta
        self.refrac = refrac
        self.tau_theta = tau_theta
        self.theta_increment = theta_increment
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_stdp = tau_stdp
        self.max_w = max_w
        self.min_w = min_w
        self.record_history = plot
        self.plot = plot
        self.device = device

        #This will set after training
        self.class_neuron_mapping = None
        self.neuron_class_mapping = None

        # Initialize neurons and synapses
        self.create_neurons()
        self.create_synapses()

        self._learning = False
        
    @property
    def learning(self):
        return self._learning

    @learning.setter
    def learning(self, value):
        self._learning = value
        # Update the homeostasis of the hidden_layer neurons
        self.hidden_layer.homeostasis = value
        print(f"Learning: {value}")
        print(f"Homeostasis: {self.hidden_layer.homeostasis}")
    
    def __repr__(self):
        return (
            f"Network("
            f"T={self.T}, "
            f"hidden_size={self.hidden_size}, "
            f"tau={self.tau}, "
            f"R={self.R}, "
            f"scale={self.scale}, "
            f"dt={self.dt}, "
            f"V_rest={self.V_rest}, "
            f"theta={self.theta}, "
            f"refrac={self.refrac}, "
            f"tau_theta={self.tau_theta}, "
            f"theta_increment={self.theta_increment}, "
            f"A_plus={self.A_plus}, "
            f"A_minus={self.A_minus}, "
            f"tau_stdp={self.tau_stdp}, "
            f"max_w={self.max_w}, "
            f"min_w={self.min_w}, "
            f"record_history={self.record_history}, "
            f"device='{self.device}'"
            f")"
        )
    
    def clear_neurons(self):
        """Reset the internal state of all neurons in the network."""
        self.hidden_layer.reset()
        
        if self.record_history:
            self.hidden_layer.V_history.clear()
            self.hidden_layer.theta_history.clear()
            
            self.synapse_input_hidden.w_history.clear()
    
    def create_neurons(self):
        self.input_layer = ReceptiveField(input_size=self.input_size,
                                          record_history=self.record_history, 
                                          device='cuda')

        # Initialize input layer with CurrentLIF neurons (one per pixel)
        self.hidden_layer = CurrentLIF(V_rest = self.V_rest,
                                       scale = self.scale,
                                       theta = self.theta, 
                                       refrac = self.refrac, 
                                       tau_theta = self.tau_theta, 
                                       theta_increment = self.theta_increment, 
                                       record_history = self.record_history,
                                       batch_size = self.hidden_size,
                                       device = self.device)


    def create_synapses(self):
        # Synapses from input to hidden layer
        self.synapse_input_hidden = Synapse(pre_neuron = self.input_layer, 
                                            post_neuron = self.hidden_layer, 
                                            A_plus = self.A_plus,
                                            A_minus = self.A_minus,
                                            tau_stdp = self.tau_stdp, 
                                            max_w = self.max_w,
                                            min_w = self.min_w, 
                                            record_history = self.record_history, 
                                            device = self.device)

    
    def forward(self, image):
        self.input_layer.compute_potential(image, encoding='isi', time_steps=self.T)
        count_spikes = torch.zeros(self.hidden_size, device = self.device)
        
        for t in range(self.T):
            self.input_layer.fire(t) #this will generate spikes for the input layer

            # Step 3: Update hidden neurons
            self.synapse_input_hidden.transmit()
            self.hidden_layer.update()
            count_spikes += self.hidden_layer.spiked #cumulative sum of spikes
            
            if self._learning:
                self.synapse_input_hidden.update_stdp()

        if self.plot:
            self.plot_activity()

        self.clear_neurons() #clear neurons for next iteration

        return count_spikes
    
    def plot_activity(self):
        self.synapse_input_hidden.plot('input-hidden synapse')


def network_simulation():
    T = 10

    image = torch.tensor([[255, 255]], dtype=torch.float32)

    net = Network(T=T, input_size=2, hidden_size=2, scale=1, plot=True)
    net.learning = True

    outs = net.forward(image)
    print(f"Output Spike Counts", outs)

    print("Final Theta Values:", net.hidden_layer.theta)

if __name__ == '__main__':
    network_simulation()
