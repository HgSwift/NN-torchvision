import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def get_data_loader(batch_size):
    """Build a data loader from training images of MNIST."""

    # Some pre-processing
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    # This will download MNIST to a 'data' folder where you run this code
    train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # Build the data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader


class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self):
        """
        Define the layers in the discriminator network.
        """
        super().__init__()
        # Your code here
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.leakyRelu = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=4, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        """
        Define forward pass in the discriminator network.

        Arguments:
            x: A tensor with shape (batch_size, 1, 32, 32).

        Returns:
            A tensor with shape (batch_size), predicting the probability of
            each example being a *real* image. Values in range (0, 1).
        """
        # Your code here        
        y = self.conv1(x)
        y = self.leakyRelu(y)
        y = self.maxpool1(y)
        y = self.conv2(y)
        y = self.leakyRelu(y)
        y = self.maxpool1(y)
        y = self.conv3(y)
        y = self.leakyRelu(y)
        y = self.maxpool1(y)
        y = self.conv4(y)
        y = self.leakyRelu(y)
        y = self.maxpool2(y)
        #y = y.view(y.shape()[0], y.shape()[2], y.shape()[3], y.shape()[1])
        y = self.linear(y.view(y.shape[0], y.shape[2], y.shape[3], y.shape[1]))
        y = self.sig(y)
        return(y.view(y.shape[0]))


class Generator(nn.Module):
    """Generator network."""

    def __init__(self,):
        """
        Define the layers in the generator network.
        """
        super().__init__()
        # Your code here
        self.leakyRelu = nn.LeakyReLU()
        self.convT1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0)
        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.convT3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.convT4 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        """
        Define forward pass in the generator network.

        Arguments:
            z: A tensor with shape (batch_size, 128).

        Returns:
            A tensor with shape (batch_size, 1, 32, 32). Values in range (-1, 1).
        """
        # Your code here
        y = z.view(z.shape[0],128,1,1)
        y = self.convT1(y)
        y = self.leakyRelu(y)
        y = self.convT2(y)
        y = self.leakyRelu(y)
        y = self.convT3(y)
        y = self.leakyRelu(y)
        y = self.convT4(y)
        y = self.tanh(y)
        return(y)


class GAN(object):
    """Generative Adversarial Network."""

    def __init__(self):
        # This will use GPU if available - don't worry if you only have CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = get_data_loader(batch_size=64)
        self.D = Discriminator().to(self.device)
        self.G = Generator().to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.001)

    def calculate_V(self, x, z):
        """
        Calculate the optimization objective for discriminator and generator
        as specified.

        Arguments:
            x: A tensor representing the real images,
                with shape (batch_size, 1, 32, 32).
            z: A tensor representing the latent variable (randomly sampled
                from Gaussian), with shape (batch_size, 128).

        Return:
            A tensor with only one element, representing the objective V.
        """
        # Your code here
        batch_size = x.shape[0]
        D1 = torch.sum(torch.log(self.D(x))) * (1/batch_size)
        G1 = torch.sum(torch.log(1-(self.D(self.G(z))))) * (1/batch_size)
        return(D1+G1)

    def train(self, epochs):
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))

            # Data loader will also return the labels of the digits (_), but we will not use them
            for iteration, (x, _) in enumerate(self.data_loader):
                x = x.to(self.device)
                z = torch.randn(64, 128).to(self.device)

                # Train the discriminator
                # We want to maximize V <=> minimize -V
                self.D_optimizer.zero_grad()
                D_target = -self.calculate_V(x, z)
                D_target.backward()
                self.D_optimizer.step()

                # Train the generator
                # We want to minimize V
                self.G_optimizer.zero_grad()
                G_target = self.calculate_V(x, z)
                G_target.backward()
                self.G_optimizer.step()

                if iteration % 100 == 0:
                    print('Iteration {}, V={}'.format(iteration, G_target.item()))

            self.visualize('Epoch {}.png'.format(epoch))

    def visualize(self, save_path):
        # Let's generate some images and visualize them
        z = torch.randn(64, 128).to(self.device)
        fake_images = self.G(z)
        save_image(fake_images, save_path)


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=10)
