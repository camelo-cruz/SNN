import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ReceptiveField:
    def __init__(self, input_size, record_history=False, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = input_size
        self.record_history = record_history
        self.spike_history = [] if record_history else None
        self.spiked = None
        self.encoded_image = None

    def compute_potential(self, image, encoding='poisson', time_steps=100):
        self.time_steps = time_steps
        image = image.view(-1).to(self.device)  # Flatten the image to a vector (28*28)
        max_pixel = torch.max(image)
        epsilon = 1e-6  # To avoid division by zero
        self.normalized_image = image / (max_pixel + epsilon)  # Normalize pixel values
        self.encoded_image = torch.zeros((len(image), self.time_steps), device=self.device)

        if encoding == 'isi':
            for i, intensity in enumerate(self.normalized_image):
                if intensity > 0:
                    isi = 1.0 / (intensity + epsilon)  # Calculate the inter-spike interval
                    spike_times = torch.arange(0, self.time_steps, isi, device=self.device).long()
                    spike_times = spike_times[spike_times < self.time_steps]  # Ensure spike times are within bounds
                    self.encoded_image[i, spike_times] = 1
        elif encoding == 'poisson':
            for i, intensity in enumerate(self.normalized_image):
                spike_times = torch.bernoulli(torch.full((self.time_steps,), intensity, device=self.device)).long()
                self.encoded_image[i] = spike_times
    
    def fire(self, t):
        self.spiked = self.encoded_image[:, t]
        if self.record_history:
            self.spike_history.append(self.spiked)

    def plot(self, title):
        fig, ax = plt.subplots()
        im = ax.imshow(self.encoded_image[:, 0].cpu().reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        
        def update(t):
            im.set_array(self.encoded_image[:, t].cpu().reshape(28, 28))
            ax.set_title(f'Timestep {t}')
        
        ani = animation.FuncAnimation(fig, update, frames=self.time_steps, repeat=False)
        
        # Ensure the save path includes a valid video file extension
        save_path = os.path.join(os.getcwd(), 'plots', f"{title}.mp4")  # Added .mp4 extension
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the animation as an MP4 video file
        ani.save(save_path, writer='ffmpeg', fps=10)
        print('Saved animation to', save_path)
        plt.close()


if __name__ == '__main__':
    # Define transformations and dataset for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Load an example image
    for images, labels in train_loader:
        example_image = images[0]  # Take the first image from the batch
        break

    T = 10  # Set the number of timesteps

    # Initialize the receptive field
    receptive_field = ReceptiveField(time_steps=T, device='cuda', record_history=True)

    # Compute the potential of the image and encode it
    receptive_field.compute_potential(example_image, encoding='isi')

    for t in range(T):
        receptive_field.fire(t)
        print(receptive_field.spiked)


    # Check encoded_image dimensions and values
    print(f"Encoded image dimensions: {receptive_field.encoded_image.shape}")

    # Plot and save the spike activity as a video
    receptive_field.plot('Encoded Image')
