r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
The jacobian is calculated by the partial derivatives of the output in respect to the input. 
Input = (128,1024)
Output = (128,2048)
Therefor the shape of the Jacobian matrix would be (1024,2048)
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.01
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.01
    lr_momentum = 0.01
    lr_rmsprop = 0.01
    reg = 0.01
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd = 1e-1
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
Yes this is possible, because cross-entropy loss measures how confident you are about a predication and accuracy measures if the prediction was correct. Therefor, you can have a high accuracy and not be as confident about it resulting in High accuracy and High loss.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
1. 
(n*m*l+1)*k = number of parameters for convolutional layer. where, 
l = input channels
k = output channels
n*m = filter size
take the sum from each layer to get the total amount of parameters. 

Left Diagram (regular block):
(3*3*64+1)*64 = 36,928 
36,928 Parameters * 2 Layers = 73,856 Parameters

Right Diagram (bottleneck block):
(1*1*256+1)*64 = 16,448 
(3*3*64+1)*64 = 36,928
(1*1*64+1)*256 = 16,640

Then we add up all the layers to get a total of: 16,448 + 36,928 + 16,640 = 70,016 Parameters

2.
Floating Point Operations (FLOPs)= number_of_parameters * (H * W) where,
H and W are the height and width of a convolution kernel

Left Diagram (regular block):
FLOPs: 73,856 Parameters * (H*W)

Right Diagram (bottleneck block):
FLOPs: 70,016 Parameters * (H*W)

3. NEED TO ANSWER
(1) spatially (within feature maps):  Within feature maps the ResBlock have a better ability to combine the input, as 
they deal with no dimensional reduction as in the BottleNeck case, who forces the input and output features to be 1. 
(2) across feature maps - the bottle neck dimensional reduction allows for better information transfer between feature
maps, as the input is more compact. 

"""

part3_q2 = r"""
**Your answer:**
Explain the effect of depth on the accuracy. What depth produces the best results and why do you think that's the case?
Adding more depth increases the accuracy up until a certain point and then if you keep adding it will end up decreasing the accuracy. 
By looking at the graphs we can see that best accuracy was acheived when L=4 and K=64

Were there values of L for which the network wasn't trainable? what causes this? Suggest two things which may be done to resolve it at least partially.
The network was not trainable when L=8,16 this can occur beacause of the unstable gradient problem when the depth becomes too large. As more layers are used in the network then the gradients loss function approaches zero, making it hard to train the netowrk. Two ways to partially resolve this problem is by using use other activation functions (ReLU) or by using residual networks.
"""


part3_q3 = r"""
**Your answer:**
In experment 1.1 and 1.2 we can see that both we can see that both have similiar behaviors. As L=8 we see that the network became untrainable in both experiments. 
When L=4 we can see in both experiments that it performs better with a larger filter size per layer. When L=2 we can see that largest amount of filters isn't necessarily the best accuracy by looking at K=128 and K=256. Overall by increasing K and L to a certain amount will help improve the accuracy. 
"""

part3_q4 = r"""
**Your answer:**
We can see that as L increased too much it started to lose it's accuracy and became untrainable. As L=3 we observe that the accuracy dropped by about 20% and as L=4 it became untrainable. The best accuracy was when  L=2.
"""

part3_q5 = r"""
**Your answer:**
In this experiment we can see that as L increases it is still trainable, in contrast to the other experiments that we ran on a regular cnn when L=8,16 it became untrainable.  
We can also see that when L=2 and K=[64-128-256] then the best accuracy is reached in this experiment and in 1.3

"""

part3_q6 = r"""
**Your answer:**

"""
# ==============

