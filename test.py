import unittest
import torch
from Receptive_Field import Receptive_Field
from Neuron import CurrentLIF
from Synapse import Synapse
import torch.nn.functional as F

ReceptiveField = Receptive_Field()
CurrentLIF = CurrentLIF()
Synapse = Synapse()

class TestContrastiveNetwork(unittest.TestCase):
    def setUp(self):
        # Parameters for the network.
        self.lr = 0.01
        self.gamma = 0.9
        self.decay = 0.1
        self.spikes = 0.5
        self.T = 5
        self.input_size = 10
        self.hidden_size = [4, 2]  # Two hidden layers.
        self.tau = 10.0
        self.R = 1.0
        self.scale = 1.0
        self.dt = 1.0
        self.V_rest = 0.0
        self.theta = 1.0
        self.refractory_period = 2
        self.record_history = False
        self.device = 'cpu'

        # Instantiate the ContrastiveNetwork.
        self.network = ContrastiveNetwork(
            lr=self.lr,
            gamma=self.gamma,
            decay=self.decay,
            spikes=self.spikes,
            T=self.T,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            tau=self.tau,
            R=self.R,
            scale=self.scale,
            dt=self.dt,
            V_rest=self.V_rest,
            theta=self.theta,
            refractory_period=self.refractory_period,
            record_history=self.record_history,
            device=self.device
        )

    def test_weight_shapes(self):
        # Check that the feedforward weight matrices have the expected shapes.
        # Weight from input to first hidden layer:
        self.assertEqual(self.network.W[0].shape, (self.input_size, self.hidden_size[0]))
        # Weight between first and second hidden layer:
        self.assertEqual(self.network.W[1].shape, (self.hidden_size[0], self.hidden_size[1]))
        # Verify the feedback synapse weight is the transpose of the connection from hidden layer 0->1.
        # Feedback synapse from hidden layer 1 (pre) to hidden layer 0 (post):
        self.assertEqual(self.network.feedback_synapses[0].w.shape, (self.hidden_size[1], self.hidden_size[0]))

    def test_free_phase_output_shape(self):
        # Create a dummy image tensor.
        batch_size = 3
        # For the dummy receptive field, we simulate an image as a (batch_size, input_size) tensor.
        dummy_image = torch.ones((batch_size, self.input_size), device=self.device)
        # Call free_phase.
        x_free = self.network.free_phase(dummy_image)
        # Expect x_free to be a list with one element per layer + the input layer,
        # i.e. len(hidden_layers)+1.
        self.assertEqual(len(x_free), len(self.hidden_size) + 1)
        # For the input layer, x0 is computed as a sum over dimension 1 (i.e. (batch_size,) reshaped to (batch_size, 1)).
        self.assertEqual(x_free[0].shape, (batch_size, 1))
        # Check dimensions for hidden layer outputs.
        self.assertEqual(x_free[1].shape, (batch_size, self.hidden_size[0]))
        self.assertEqual(x_free[2].shape, (batch_size, self.hidden_size[1]))

    def test_clamped_phase_output_shape(self):
        # Create a dummy image tensor.
        batch_size = 3
        dummy_image = torch.ones((batch_size, self.input_size), device=self.device)
        # Create dummy target labels (for 10 classes).
        dummy_target = torch.randint(low=0, high=10, size=(batch_size,), device=self.device)
        # Call clamped_phase.
        x_clamped = self.network.clamped_phase(dummy_image, dummy_target)
        # Check the length of the output list.
        self.assertEqual(len(x_clamped), len(self.hidden_size) + 1)
        # Validate shapes: input layer output shape and hidden layers.
        self.assertEqual(x_clamped[0].shape, (batch_size, 1))
        self.assertEqual(x_clamped[1].shape, (batch_size, self.hidden_size[0]))
        self.assertEqual(x_clamped[2].shape, (batch_size, self.hidden_size[1]))

    def test_contrastive_update_changes_weights(self):
        # Create dummy free and clamped phase outputs with the required dimensions.
        batch_size = 4
        x_free = [
            torch.ones((batch_size, 1), dtype=torch.int32, device=self.device),
            torch.ones((batch_size, self.hidden_size[0]), dtype=torch.int32, device=self.device),
            torch.ones((batch_size, self.hidden_size[1]), dtype=torch.int32, device=self.device)
        ]
        x_clamped = [
            torch.ones((batch_size, 1), dtype=torch.int32, device=self.device) * 2,
            torch.ones((batch_size, self.hidden_size[0]), dtype=torch.int32, device=self.device) * 2,
            torch.ones((batch_size, self.hidden_size[1]), dtype=torch.int32, device=self.device) * 2
        ]
        # Save copies of the original weights.
        original_W0 = self.network.W[0].clone()
        original_W1 = self.network.W[1].clone()
        # Perform a contrastive update.
        self.network.contrastive_update(x_free, x_clamped)
        # Verify that the weight matrices have changed.
        self.assertFalse(torch.equal(original_W0, self.network.W[0]))
        self.assertFalse(torch.equal(original_W1, self.network.W[1]))
        # Verify that the synapse weights have been updated.
        self.assertTrue(torch.equal(self.network.feedforward_synapses[0].w, self.network.W[0]))
        self.assertTrue(torch.equal(self.network.feedforward_synapses[1].w, self.network.W[1]))
        self.assertTrue(torch.equal(self.network.feedback_synapses[0].w, self.network.W[1].t()))

if __name__ == "__main__":
    unittest.main()
