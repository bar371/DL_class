import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autograd
from typing import Callable

from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

CLIP_VALUE = 0.01

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        in_channels = self.in_size[0]
        feature_mapping = 64
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, feature_mapping, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_mapping, feature_mapping * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_mapping * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_mapping * 2, feature_mapping * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_mapping * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_mapping * 4, feature_mapping * 8, 4, 2, 1, bias=False),
            nn.Dropout(0.5),
            nn.BatchNorm2d(feature_mapping * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_mapping * 8, 1, 4, 1, 0, bias=False),
            )

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        y = self.discriminator(x).reshape((x.shape[0],1))
        return y


class SNDiscriminator(nn.Module):
    def __init__(self):
        self.nowg = 16
        self.leak = 0.1
        super().__init__()
        self.sndiscrimentaor = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 3, stride=1, padding=(1, 1))),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1))),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1))),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1))),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1))),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1))),
            nn.LeakyReLU(self.leak, inplace=True),
        )
        self.fc = spectral_norm(nn.Linear(8192, 1))

    def forward(self, x):
        z = self.sndiscrimentaor(x)
        return self.fc(z.view(-1, 8192))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(z_dim, featuremap_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(featuremap_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(featuremap_size * 8, featuremap_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featuremap_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( featuremap_size * 4, featuremap_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featuremap_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( featuremap_size * 2, featuremap_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featuremap_size),
            nn.ReLU(True),
            nn.ConvTranspose2d( featuremap_size, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        insts = torch.randn(n, self.z_dim, 1, 1).to(device)
        if with_grad:
            samples = self.forward(insts)
        else:
            with torch.no_grad():
                samples = self.forward(insts)
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """

        z = z.reshape(z.shape[0], z.shape[1], 1, 1)
        x = self.generator(z)
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0

    # ====== YOUR CODE: ======
    noise_side = label_noise/2
    m = torch.distributions.uniform.Uniform(data_label-noise_side, data_label+noise_side)
    noise = m.sample(sample_shape=y_data.shape).to(y_data.device)
    data_cnt = BCEWithLogitsLoss()
    loss_data = data_cnt(y_data, noise)
    m = torch.distributions.uniform.Uniform(-noise_side, noise_side)
    noise = m.sample(sample_shape=y_generated.shape).to(y_generated.device)
    gen_cnt = BCEWithLogitsLoss()
    loss_generated = gen_cnt(y_generated, noise)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0

    if data_label == 1:
        labels = torch.ones(y_generated.shape, device=y_generated.device, dtype=torch.float)
    else:
        labels = torch.zeros(y_generated.shape, device=y_generated.device, dtype=torch.float)
    loss_f = BCEWithLogitsLoss()
    loss = loss_f(y_generated, labels)
    return loss

# During discriminator forward-backward-update
def dsc_wgan_loss(y_data, y_generated):
    return torch.mean(y_generated) - torch.mean(y_data)



def gen_wgan_loss(y_generated):
    return -torch.mean(y_generated)


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
    critic=int,
    mode='vanilla'
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    Mode - select pure DC-GAN training or weight clapping and power iteration for discrimenator by WGAN
    :return: The discriminator and generator losses.
    """

    dsc_model.train(True)
    dsc_model.zero_grad()
    gen_samples = gen_model.sample(len(x_data),with_grad=False)
    dsc_loss = dsc_loss_fn(dsc_model(x_data), dsc_model(gen_samples))
    dsc_loss.backward()
    dsc_optimizer.step()

    if mode == 'wgan' or mode == 'wgan_gd': # train by wgan
        for p in dsc_model.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        if critic:
            gen_model.train(True)
            gen_model.zero_grad()
            gen_samples = gen_model.sample(len(x_data), with_grad=True)
            gen_loss = gen_loss_fn(dsc_model(gen_samples))
            gen_loss.backward()
            gen_optimizer.step()
            if mode == 'wgan_gd':
                gp = calc_gradient_penalty(dsc_model, autograd.Variable(x_data), autograd.Variable(gen_samples))
                gp.backward()
            return dsc_loss.item(), gen_loss.item()
        else:
            if mode == 'wgan_gd':
                gp = calc_gradient_penalty(dsc_model, autograd.Variable(x_data), autograd.Variable(gen_samples))
                gp.backward()
            return dsc_loss.item(), None

    elif mode == 'vanilla': # vanilla
        gen_model.train(True)
        gen_model.zero_grad()
        gen_samples = gen_model.sample(len(x_data), with_grad=True)
        gen_loss = gen_loss_fn(dsc_model(gen_samples))
        gen_loss.backward()
        gen_optimizer.step()
        return dsc_loss.item(), gen_loss.item()

    # ========================



def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"
    pickle.dump(gen_losses, open(checkpoint_file[:-2] + 'gen_losses','wb'))
    pickle.dump(dsc_losses, open(checkpoint_file[:-2] + 'dcs_losses','wb'))
    torch.save(gen_model, open(checkpoint_file, 'wb'))
    saved = True
    # ========================

    return saved

# Note - gradient penalty calcuation was taken from the GD penalty article writer's github
# https://github.com/caogang/wgan-gp/blob/ae47a185ed2e938c39cf3eb2f06b32dc1b6a2064/gan_mnist.py#L129
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(64, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda('cuda')
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda('cuda')
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda('cuda'),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty