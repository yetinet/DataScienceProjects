import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import imageio

# %matplotlib inline

# Load VGG 19 model
# I think this gets the weights from each layer in addition to other stuff
# model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
# print(model)

# read in content image
# content_image = scipy.misc.imread("images/louvre.jpg")
# imshow(content_image)

# Create Content Cost Function
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, [m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [m, n_H * n_W, n_C])

    # compute the cost with tensorflow (≈1 line)
    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.squared_difference(a_C_unrolled, a_G_unrolled))
    ### END CODE HERE ###

    return J_content

# Test Content Function
# Expected Output is J_content = 6.76559
# tf.reset_default_graph()
#
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     J_content = compute_content_cost(a_C, a_G)
#     print("J_content = " + str(J_content.eval()))


# Loading the STYLE Image
# style_image = scipy.misc.imread("images/monet_800600.jpg")
# imshow(style_image)


# Functions for the Style Cost
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    ### START CODE HERE ### (≈1 line)
    GA = tf.matmul(A, A, transpose_b=True)
    ### END CODE HERE ###

    return GA

# Gram Matrix that computes similarities between matrices
# G_a = A(A^T)
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, [m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [m, n_H * n_W, n_C])

    # compute the cost with tensorflow (≈1 line)
    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.squared_difference(a_C_unrolled, a_G_unrolled))
    ### END CODE HERE ###

    return J_content

# Tet Gram Function
# EXPECTED OUTPUT
# GA = [[  6.42230511  -4.42912197  -2.09668207]
#  [ -4.42912197  19.46583748  19.56387138]
#  [ -2.09668207  19.56387138  20.6864624 ]]
# tf.reset_default_graph()
#
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     A = tf.random_normal([3, 2 * 1], mean=1, stddev=4)
#     GA = gram_matrix(A)
#
#     print("GA = " + str(GA.eval()))



# Put Style Cost Together now
# GRADED FUNCTION: compute_layer_style_cost

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, (n_H * n_W, n_C)))
    a_G = tf.transpose(tf.reshape(a_G, (n_H * n_W, n_C)))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum(tf.squared_difference(GS, GG)) / (4 * n_C ** 2 * (n_H * n_W) ** 2)

    ### END CODE HERE ###

    return J_style_layer


# Test Style Cost Function
# EXPECTED OUTPUT
# J_style_layer = 9.19028
# tf.reset_default_graph()
#
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     J_style_layer = compute_layer_style_cost(a_S, a_G)
#
#     print("J_style_layer = " + str(J_style_layer.eval()))


# Weight Costs From different layers of the Styles image Activations
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


# Total Style Cost with Wieghts on layers
def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


# total_cost
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    ### START CODE HERE ### (≈1 line)
    J = alpha * J_content + beta * J_style
    ### END CODE HERE ###

    return J

# Test Total Cost
# EXPECTED OUTPUT
# J = 35.34667875478276
# tf.reset_default_graph()
#
# with tf.Session() as test:
#     np.random.seed(3)
#     J_content = np.random.randn()
#     J_style = np.random.randn()
#     J = total_cost(J_content, J_style)
#     print("J = " + str(J))



# Here's what the program will have to do:

# Create an Interactive Session
# Load the content image
# Load the style image
# Randomly initialize the image to be generated
# Load the VGG16 model
# Build the TensorFlow graph:
# Run the content image through the VGG16 model and compute the content cost
# Run the style image through the VGG16 model and compute the style cost
# Compute the total cost
# Define the optimizer and the learning rate
# Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.
# Lets go through the individual steps in detail.



# 1. Start Interactive Session
# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()




# 2a. Let's load, reshape, and normalize our "content" image
content_image = scipy.misc.imread(content_image)

# 2b. Let's load, reshape and normalize our "style" image:
style_image = scipy.misc.imread(style_image)
# style_image = match_image_sizes(content_image, style_image)

content_image = reshape_and_normalize_image(content_image)
style_image = reshape_and_normalize_image(style_image)



# 3. Now, we initialize the "generated" image as a noisy image created
#    from the content_image. By initializing the pixels of the generated
#    image to be mostly noise but still slightly correlated with the content
#    image, this will help the content of the "generated" image more rapidly
#    match the content of the "content" image.

# imshow(generated_image[0])
generated_image = generate_noise_image(content_image)





# 4. Load VGG 16 model
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")





# 5. To get the program to compute the content cost, we will now assign a_C and a_G to be the appropriate hidden layer activations. We will use layer conv4_2 to compute the content cost. The code below does the following:
#
# Assign the content image to be the input to the VGG model.
# Set a_C to be the tensor giving the hidden layer activation for layer "conv4_2".
# Set a_G to be the tensor giving the hidden layer activation for the same layer.
# Compute the content cost using a_C and a_G.
# Assign the content image to be the input of the VGG model.
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)





# At this point, a_G is a tensor and hasn't been evaluated. It will be evaluated
# and updated at each iteration when we run the Tensorflow graph in model_nn() below.
# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)


# Compute Total Cost
# changed from 10 to 20
J = total_cost(J_content, J_style, 10, 40)


# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)


# Implement the model_nn() function which initializes
# the variables of the tensorflow graph, assigns the input image
# (initial generated image) as the input of the VGG16 model and
# runs the train_step for a large number of steps.
def model_nn(sess, input_image, num_iterations=200):
    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    sess.run(tf.global_variables_initializer())
    ### END CODE HERE ###

    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    sess.run(model['input'].assign(input_image))
    ### END CODE HERE ###

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        _ = sess.run(train_step)
        ### END CODE HERE ###

        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image



 # Run the following cell to generate an artistic image. It should take
    # about 3min on CPU for every 20 iterations but you start observing
    # attractive results after ≈140 iterations. Neural Style Transfer is
    # generally trained using GPUs.


model_nn(sess, generated_image)