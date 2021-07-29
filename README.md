#  Universita of Haifa CV - Visualizations for basic NCC
## About:
Visualizations for basic NCC.

1. MNIST is a dataset of 70,000 grayscale hand-written digits (0 through 9). 60,000 of these are training images. 10,000 are a held out test set.

CIFAR-10 is a dataset of 60,000 color images (32 by 32 resolution) across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The train/test split is 50k/10k.

   - Use matplotlib and ipython notebook's visualization capabilities to display one train and one test image of each class, for each of the two datasets.
![CIFAR-10_result](./output/CIFAR-10_result.png?raw=true)
![MINT_Visualization_results](./output/MINT_Visualization_results.png?raw=true)

2. Start by running the training on MNIST. By default if you run this notebook successfully, it will train on MNIST.

This will initialize a single layer model train it on the 60,000 MNIST training images for 10 epochs (passes through the training data).

The loss function cross_entropy computes a Logarithm of the Softmax on the output of the neural network, and then computes the negative log-likelihood w.r.t. the given target.

The default values for the learning rate, batch size and number of epochs are given in the "options" cell of this notebook. Unless otherwise specified, use the default values throughout this assignment.

Note the decrease in training loss and corresponding decrease in validation errors.

  a) Add code to plot out the network weights as images (one for each output, of size 28 by 28) after the last epoch. (Hint threads: #1 #2 )
  b) Reduce the number of training examples to just 50.
  
3. a) Add an extra layer to the network with 1000 hidden units and a tanh non-linearity. [Hint: modify the Net class] and train the model for 10 epochs.
   b) Now set the learning rate to 10 and retrain.
   
   
4. To change over to the CIFAR-10 dataset, change the options cell's dataset variable to 'cifar10'.

Create a convolutional network with the following architecture:
```
Convolution with 5 by 5 filters, 16 feature maps + Tanh nonlinearity.
2 by 2 max pooling (non-overlapping).
Convolution with 5 by 5 filters, 128 feature maps + Tanh nonlinearity.
2 by 2 max pooling (non-overlapping).
Flatten to vector.
Linear layer with 64 hidden units + Tanh nonlinearity.
Linear layer to 10 output units.
- Train it for 20 epochs on the CIFAR-10 training set and Visualize the first layer filters. 
```
