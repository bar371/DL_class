r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""



# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

"""

part2_q2 = r"""
**Your answer:**

"""

part2_q3 = r"""
**Your answer:**


"""

part2_q4 = r"""
**Your answer:**

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="SGD",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 4
    hypers['discriminator_optimizer']['lr'] = 0.0002
    # hypers['discriminator_optimizer']['betas'] = (0.5, 0.999)
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['betas'] = (0.5, 0.999)
    hypers['label_noise'] = 0.25
    hypers['data_label'] = 1
    hypers['z_dim'] = 100
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
We would like to maintain the gradients when sampling form the GAN while we train the generator, as the loss function
for the generator is calculated by the images sampled, in order for to improve the generator during training we must
keep the gradients of these samples, adapting the generator weights during backprop step.
 
However, when we train the Discriminator we would not want to keep the gradients of the samples, 
but only use their loss to calculate the total loss for the Discriminator  
and in the backprop step affect only the Discriminator. 
"""

part3_q2 = r"""
**Your answer:**
A. The generator loss is not enough on its on, as it has direct connection to the discriminator loss.
When the generator loss is low and the Discriminator loss is relatively high it means the generator is able to 'fool'
the discriminator. As such, when we evaluate the training process we must look at both the discriminator and 
generator loss.

B. Having the generator loss decreasing while the discriminator loss not changing implies that the generator is 
improving faster than the discriminator. As we have seen on several articles a way to approch this problem is to
give the discriminator more training time than the generator (And we have found that this is advised in general).
"""

part3_q3 = r"""
**Your answer:**

"""

# ==============
