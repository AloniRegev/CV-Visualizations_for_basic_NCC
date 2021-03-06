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

  - Add code to plot out the network weights as images (one for each output, of size 28 by 28) after the last epoch.
  
  ![filter_result](./output/filter_result.png?raw=true)
  
3. 
   - Add an extra layer to the network with 1000 hidden units and a tanh non-linearity. [Hint: modify the Net class] and train the model for 10 epochs.
   ```
   Learning Rate:  0.01
   Train Epoch: 1 [0/60000 (0%)]	Loss: 2.349562
   Train Epoch: 1 [6400/60000 (11%)]	Loss: 0.774654
   Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.575926
   Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.367759
   Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.315608
   Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.252229
   Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.282357
   Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.422812
   Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.310872
   Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.384680
   Train Epoch: 2 [0/60000 (0%)]	Loss: 0.486581
   Train Epoch: 2 [6400/60000 (11%)]	Loss: 0.317732
   Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.406053
   Train Epoch: 2 [19200/60000 (32%)]	Loss: 0.362434
   Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.454347
   Train Epoch: 2 [32000/60000 (53%)]	Loss: 0.406068
   Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.420542
   Train Epoch: 2 [44800/60000 (75%)]	Loss: 0.298357
   Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.418264
   Train Epoch: 2 [57600/60000 (96%)]	Loss: 0.249776
   Train Epoch: 3 [0/60000 (0%)]	Loss: 0.204722
   Train Epoch: 3 [6400/60000 (11%)]	Loss: 0.226249
   Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.163245
   Train Epoch: 3 [19200/60000 (32%)]	Loss: 0.174127
   Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.253093
   Train Epoch: 3 [32000/60000 (53%)]	Loss: 0.243791
   Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.285285
   Train Epoch: 3 [44800/60000 (75%)]	Loss: 0.328761
   Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.422124
   Train Epoch: 3 [57600/60000 (96%)]	Loss: 0.262448
   Train Epoch: 4 [0/60000 (0%)]	Loss: 0.292104
   Train Epoch: 4 [6400/60000 (11%)]	Loss: 0.330583
   Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.264967
   Train Epoch: 4 [19200/60000 (32%)]	Loss: 0.215000
   Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.195254
   Train Epoch: 4 [32000/60000 (53%)]	Loss: 0.393793
   Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.187265
   Train Epoch: 4 [44800/60000 (75%)]	Loss: 0.223670
   Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.156934
   Train Epoch: 4 [57600/60000 (96%)]	Loss: 0.144040
   Train Epoch: 5 [0/60000 (0%)]	Loss: 0.120252
   Train Epoch: 5 [6400/60000 (11%)]	Loss: 0.138702
   Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.278181
   Train Epoch: 5 [19200/60000 (32%)]	Loss: 0.100742
   Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.157441
   Train Epoch: 5 [32000/60000 (53%)]	Loss: 0.243193
   Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.187095
   Train Epoch: 5 [44800/60000 (75%)]	Loss: 0.085921
   Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.322485
   Train Epoch: 5 [57600/60000 (96%)]	Loss: 0.311115
   Train Epoch: 6 [0/60000 (0%)]	Loss: 0.216807
   Train Epoch: 6 [6400/60000 (11%)]	Loss: 0.346529
   Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.215285
   Train Epoch: 6 [19200/60000 (32%)]	Loss: 0.169845
   Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.213310
   Train Epoch: 6 [32000/60000 (53%)]	Loss: 0.146957
   Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.153669
   Train Epoch: 6 [44800/60000 (75%)]	Loss: 0.176857
   Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.129671
   Train Epoch: 6 [57600/60000 (96%)]	Loss: 0.113569
   Train Epoch: 7 [0/60000 (0%)]	Loss: 0.160645
   Train Epoch: 7 [6400/60000 (11%)]	Loss: 0.196573
   Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.129174
   Train Epoch: 7 [19200/60000 (32%)]	Loss: 0.154157
   Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.119950
   Train Epoch: 7 [32000/60000 (53%)]	Loss: 0.134971
   Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.187109
   Train Epoch: 7 [44800/60000 (75%)]	Loss: 0.192786
   Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.261110
   Train Epoch: 7 [57600/60000 (96%)]	Loss: 0.125060
   Train Epoch: 8 [0/60000 (0%)]	Loss: 0.388325
   Train Epoch: 8 [6400/60000 (11%)]	Loss: 0.312638
   Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.159065
   Train Epoch: 8 [19200/60000 (32%)]	Loss: 0.256635
   Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.087121
   Train Epoch: 8 [32000/60000 (53%)]	Loss: 0.220031
   Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.132193
   Train Epoch: 8 [44800/60000 (75%)]	Loss: 0.105483
   Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.354284
   Train Epoch: 8 [57600/60000 (96%)]	Loss: 0.307615
   Train Epoch: 9 [0/60000 (0%)]	Loss: 0.187346
   Train Epoch: 9 [6400/60000 (11%)]	Loss: 0.116514
   Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.097118
   Train Epoch: 9 [19200/60000 (32%)]	Loss: 0.104559
   Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.106464
   Train Epoch: 9 [32000/60000 (53%)]	Loss: 0.157133
   Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.218650
   Train Epoch: 9 [44800/60000 (75%)]	Loss: 0.103350
   Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.221722
   Train Epoch: 9 [57600/60000 (96%)]	Loss: 0.184341
   Train Epoch: 10 [0/60000 (0%)]	Loss: 0.106886
   Train Epoch: 10 [6400/60000 (11%)]	Loss: 0.101933
   Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.157427
   Train Epoch: 10 [19200/60000 (32%)]	Loss: 0.325799
   Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.046667
   Train Epoch: 10 [32000/60000 (53%)]	Loss: 0.213049
   Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.112099
   Train Epoch: 10 [44800/60000 (75%)]	Loss: 0.084439
   Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.118232
   Train Epoch: 10 [57600/60000 (96%)]	Loss: 0.161776
   ```
   - Now set the learning rate to 10 and retrain.
   ```
   Learning Rate:  10
   Train Epoch: 1 [0/60000 (0%)]	Loss: 0.099799
   Train Epoch: 1 [6400/60000 (11%)]	Loss: 0.142202
   Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.068991
   Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.213832
   Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.126548
   Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.198695
   Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.240308
   Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.083835
   Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.142404
   Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.122735
   Train Epoch: 2 [0/60000 (0%)]	Loss: 0.220874
   Train Epoch: 2 [6400/60000 (11%)]	Loss: 0.074893
   Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.068323
   Train Epoch: 2 [19200/60000 (32%)]	Loss: 0.256015
   Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.129301
   Train Epoch: 2 [32000/60000 (53%)]	Loss: 0.081226
   Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.173207
   Train Epoch: 2 [44800/60000 (75%)]	Loss: 0.119827
   Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.348733
   Train Epoch: 2 [57600/60000 (96%)]	Loss: 0.245968
   Train Epoch: 3 [0/60000 (0%)]	Loss: 0.089811
   Train Epoch: 3 [6400/60000 (11%)]	Loss: 0.241800
   Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.199676
   Train Epoch: 3 [19200/60000 (32%)]	Loss: 0.239204
   Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.116335
   Train Epoch: 3 [32000/60000 (53%)]	Loss: 0.157505
   Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.060089
   Train Epoch: 3 [44800/60000 (75%)]	Loss: 0.249702
   Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.079848
   Train Epoch: 3 [57600/60000 (96%)]	Loss: 0.076228
   Train Epoch: 4 [0/60000 (0%)]	Loss: 0.150908
   Train Epoch: 4 [6400/60000 (11%)]	Loss: 0.170532
   Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.146252
   Train Epoch: 4 [19200/60000 (32%)]	Loss: 0.094842
   Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.026517
   Train Epoch: 4 [32000/60000 (53%)]	Loss: 0.048292
   Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.122887
   Train Epoch: 4 [44800/60000 (75%)]	Loss: 0.076952
   Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.095552
   Train Epoch: 4 [57600/60000 (96%)]	Loss: 0.199685
   Train Epoch: 5 [0/60000 (0%)]	Loss: 0.080586
   Train Epoch: 5 [6400/60000 (11%)]	Loss: 0.093457
   Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.062534
   Train Epoch: 5 [19200/60000 (32%)]	Loss: 0.099378
   Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.051448
   Train Epoch: 5 [32000/60000 (53%)]	Loss: 0.073811
   Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.131943
   Train Epoch: 5 [44800/60000 (75%)]	Loss: 0.120240
   Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.061910
   Train Epoch: 5 [57600/60000 (96%)]	Loss: 0.175359
   Train Epoch: 6 [0/60000 (0%)]	Loss: 0.196057
   Train Epoch: 6 [6400/60000 (11%)]	Loss: 0.180854
   Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.121709
   Train Epoch: 6 [19200/60000 (32%)]	Loss: 0.233118
   Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.056233
   Train Epoch: 6 [32000/60000 (53%)]	Loss: 0.165506
   Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.080685
   Train Epoch: 6 [44800/60000 (75%)]	Loss: 0.139067
   Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.049156
   Train Epoch: 6 [57600/60000 (96%)]	Loss: 0.199971
   Train Epoch: 7 [0/60000 (0%)]	Loss: 0.117044
   Train Epoch: 7 [6400/60000 (11%)]	Loss: 0.069060
   Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.093863
   Train Epoch: 7 [19200/60000 (32%)]	Loss: 0.020921
   Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.067731
   Train Epoch: 7 [32000/60000 (53%)]	Loss: 0.149903
   Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.081763
   Train Epoch: 7 [44800/60000 (75%)]	Loss: 0.111259
   Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.278563
   Train Epoch: 7 [57600/60000 (96%)]	Loss: 0.157833
   Train Epoch: 8 [0/60000 (0%)]	Loss: 0.158415
   Train Epoch: 8 [6400/60000 (11%)]	Loss: 0.096405
   Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.163108
   Train Epoch: 8 [19200/60000 (32%)]	Loss: 0.086715
   Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.137180
   Train Epoch: 8 [32000/60000 (53%)]	Loss: 0.148983
   Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.062840
   Train Epoch: 8 [44800/60000 (75%)]	Loss: 0.222841
   Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.031631
   Train Epoch: 8 [57600/60000 (96%)]	Loss: 0.116389
   Train Epoch: 9 [0/60000 (0%)]	Loss: 0.071996
   Train Epoch: 9 [6400/60000 (11%)]	Loss: 0.093425
   Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.025285
   Train Epoch: 9 [19200/60000 (32%)]	Loss: 0.149487
   Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.149241
   Train Epoch: 9 [32000/60000 (53%)]	Loss: 0.075810
   Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.098359
   Train Epoch: 9 [44800/60000 (75%)]	Loss: 0.047787
   Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.052473
   Train Epoch: 9 [57600/60000 (96%)]	Loss: 0.101620
   Train Epoch: 10 [0/60000 (0%)]	Loss: 0.100971
   Train Epoch: 10 [6400/60000 (11%)]	Loss: 0.084612
   Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.033144
   Train Epoch: 10 [19200/60000 (32%)]	Loss: 0.232668
   Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.054304
   Train Epoch: 10 [32000/60000 (53%)]	Loss: 0.186857
   Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.102578
   Train Epoch: 10 [44800/60000 (75%)]	Loss: 0.106581
   Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.022712
   Train Epoch: 10 [57600/60000 (96%)]	Loss: 0.053385
   ```
   
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
```
- Train it for 20 epochs on the CIFAR-10 training set and Visualize the first layer filters. 
```
Train Epoch: 1 [0/50000 (0%)]	Loss: 2.283492
Train Epoch: 1 [6400/50000 (13%)]	Loss: 1.680318
Train Epoch: 1 [12800/50000 (26%)]	Loss: 1.476834
Train Epoch: 1 [19200/50000 (38%)]	Loss: 1.596486
Train Epoch: 1 [25600/50000 (51%)]	Loss: 1.638269
Train Epoch: 1 [32000/50000 (64%)]	Loss: 1.344022
Train Epoch: 1 [38400/50000 (77%)]	Loss: 1.716790
Train Epoch: 1 [44800/50000 (90%)]	Loss: 1.203530
Train Epoch: 2 [0/50000 (0%)]	Loss: 1.699515
Train Epoch: 2 [6400/50000 (13%)]	Loss: 1.205850
Train Epoch: 2 [12800/50000 (26%)]	Loss: 1.134477
Train Epoch: 2 [19200/50000 (38%)]	Loss: 1.185391
Train Epoch: 2 [25600/50000 (51%)]	Loss: 1.277264
Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.271756
Train Epoch: 2 [38400/50000 (77%)]	Loss: 1.188631
Train Epoch: 2 [44800/50000 (90%)]	Loss: 1.132245
Train Epoch: 3 [0/50000 (0%)]	Loss: 1.080816
Train Epoch: 3 [6400/50000 (13%)]	Loss: 1.192045
Train Epoch: 3 [12800/50000 (26%)]	Loss: 0.983079
Train Epoch: 3 [19200/50000 (38%)]	Loss: 0.965840
Train Epoch: 3 [25600/50000 (51%)]	Loss: 0.979916
Train Epoch: 3 [32000/50000 (64%)]	Loss: 0.979454
Train Epoch: 3 [38400/50000 (77%)]	Loss: 0.989708
Train Epoch: 3 [44800/50000 (90%)]	Loss: 1.091173
Train Epoch: 4 [0/50000 (0%)]	Loss: 0.997695
Train Epoch: 4 [6400/50000 (13%)]	Loss: 1.056491
Train Epoch: 4 [12800/50000 (26%)]	Loss: 0.991583
Train Epoch: 4 [19200/50000 (38%)]	Loss: 0.964124
Train Epoch: 4 [25600/50000 (51%)]	Loss: 1.159544
Train Epoch: 4 [32000/50000 (64%)]	Loss: 0.939816
Train Epoch: 4 [38400/50000 (77%)]	Loss: 0.868747
Train Epoch: 4 [44800/50000 (90%)]	Loss: 0.960778
Train Epoch: 5 [0/50000 (0%)]	Loss: 1.229132
Train Epoch: 5 [6400/50000 (13%)]	Loss: 0.873476
Train Epoch: 5 [12800/50000 (26%)]	Loss: 0.902496
Train Epoch: 5 [19200/50000 (38%)]	Loss: 0.903972
Train Epoch: 5 [25600/50000 (51%)]	Loss: 0.887250
Train Epoch: 5 [32000/50000 (64%)]	Loss: 0.702506
Train Epoch: 5 [38400/50000 (77%)]	Loss: 0.956183
Train Epoch: 5 [44800/50000 (90%)]	Loss: 0.638321
Train Epoch: 6 [0/50000 (0%)]	Loss: 1.433439
Train Epoch: 6 [6400/50000 (13%)]	Loss: 0.716737
Train Epoch: 6 [12800/50000 (26%)]	Loss: 0.715689
Train Epoch: 6 [19200/50000 (38%)]	Loss: 0.673878
Train Epoch: 6 [25600/50000 (51%)]	Loss: 0.719808
Train Epoch: 6 [32000/50000 (64%)]	Loss: 0.612647
Train Epoch: 6 [38400/50000 (77%)]	Loss: 0.715189
Train Epoch: 6 [44800/50000 (90%)]	Loss: 0.715900
Train Epoch: 7 [0/50000 (0%)]	Loss: 0.807033
Train Epoch: 7 [6400/50000 (13%)]	Loss: 0.674326
Train Epoch: 7 [12800/50000 (26%)]	Loss: 0.779465
Train Epoch: 7 [19200/50000 (38%)]	Loss: 0.708561
Train Epoch: 7 [25600/50000 (51%)]	Loss: 0.573808
Train Epoch: 7 [32000/50000 (64%)]	Loss: 0.523819
Train Epoch: 7 [38400/50000 (77%)]	Loss: 0.542137
Train Epoch: 7 [44800/50000 (90%)]	Loss: 0.760919
Train Epoch: 8 [0/50000 (0%)]	Loss: 0.550251
Train Epoch: 8 [6400/50000 (13%)]	Loss: 0.706308
Train Epoch: 8 [12800/50000 (26%)]	Loss: 0.757858
Train Epoch: 8 [19200/50000 (38%)]	Loss: 0.727728
Train Epoch: 8 [25600/50000 (51%)]	Loss: 0.641869
Train Epoch: 8 [32000/50000 (64%)]	Loss: 0.777728
Train Epoch: 8 [38400/50000 (77%)]	Loss: 0.733861
Train Epoch: 8 [44800/50000 (90%)]	Loss: 0.672120
Train Epoch: 9 [0/50000 (0%)]	Loss: 0.718119
Train Epoch: 9 [6400/50000 (13%)]	Loss: 0.480516
Train Epoch: 9 [12800/50000 (26%)]	Loss: 0.491239
Train Epoch: 9 [19200/50000 (38%)]	Loss: 0.615624
Train Epoch: 9 [25600/50000 (51%)]	Loss: 0.412655
Train Epoch: 9 [32000/50000 (64%)]	Loss: 0.519051
Train Epoch: 9 [38400/50000 (77%)]	Loss: 0.507627
Train Epoch: 9 [44800/50000 (90%)]	Loss: 0.649787
Train Epoch: 10 [0/50000 (0%)]	Loss: 0.516378
Train Epoch: 10 [6400/50000 (13%)]	Loss: 0.577842
Train Epoch: 10 [12800/50000 (26%)]	Loss: 0.566578
Train Epoch: 10 [19200/50000 (38%)]	Loss: 0.410657
Train Epoch: 10 [25600/50000 (51%)]	Loss: 0.434930
Train Epoch: 10 [32000/50000 (64%)]	Loss: 0.707825
Train Epoch: 10 [38400/50000 (77%)]	Loss: 0.764566
Train Epoch: 10 [44800/50000 (90%)]	Loss: 0.481098
Train Epoch: 11 [0/50000 (0%)]	Loss: 0.708156
Train Epoch: 11 [6400/50000 (13%)]	Loss: 0.402770
Train Epoch: 11 [12800/50000 (26%)]	Loss: 0.392222
Train Epoch: 11 [19200/50000 (38%)]	Loss: 0.373295
Train Epoch: 11 [25600/50000 (51%)]	Loss: 0.316454
Train Epoch: 11 [32000/50000 (64%)]	Loss: 0.517509
Train Epoch: 11 [38400/50000 (77%)]	Loss: 0.445186
Train Epoch: 11 [44800/50000 (90%)]	Loss: 0.464453
Train Epoch: 12 [0/50000 (0%)]	Loss: 0.477914
Train Epoch: 12 [6400/50000 (13%)]	Loss: 0.237856
Train Epoch: 12 [12800/50000 (26%)]	Loss: 0.550171
Train Epoch: 12 [19200/50000 (38%)]	Loss: 0.354152
Train Epoch: 12 [25600/50000 (51%)]	Loss: 0.348046
Train Epoch: 12 [32000/50000 (64%)]	Loss: 0.399144
Train Epoch: 12 [38400/50000 (77%)]	Loss: 0.447808
Train Epoch: 12 [44800/50000 (90%)]	Loss: 0.494166
Train Epoch: 13 [0/50000 (0%)]	Loss: 0.357200
Train Epoch: 13 [6400/50000 (13%)]	Loss: 0.157865
Train Epoch: 13 [12800/50000 (26%)]	Loss: 0.246984
Train Epoch: 13 [19200/50000 (38%)]	Loss: 0.439672
Train Epoch: 13 [25600/50000 (51%)]	Loss: 0.373641
Train Epoch: 13 [32000/50000 (64%)]	Loss: 0.644122
Train Epoch: 13 [38400/50000 (77%)]	Loss: 0.333095
Train Epoch: 13 [44800/50000 (90%)]	Loss: 0.310345
Train Epoch: 14 [0/50000 (0%)]	Loss: 0.550556
Train Epoch: 14 [6400/50000 (13%)]	Loss: 0.297649
Train Epoch: 14 [12800/50000 (26%)]	Loss: 0.256358
Train Epoch: 14 [19200/50000 (38%)]	Loss: 0.149737
Train Epoch: 14 [25600/50000 (51%)]	Loss: 0.305810
Train Epoch: 14 [32000/50000 (64%)]	Loss: 0.377214
Train Epoch: 14 [38400/50000 (77%)]	Loss: 0.288065
Train Epoch: 14 [44800/50000 (90%)]	Loss: 0.276855
Train Epoch: 15 [0/50000 (0%)]	Loss: 0.141676
Train Epoch: 15 [6400/50000 (13%)]	Loss: 0.208273
Train Epoch: 15 [12800/50000 (26%)]	Loss: 0.128053
Train Epoch: 15 [19200/50000 (38%)]	Loss: 0.163837
Train Epoch: 15 [25600/50000 (51%)]	Loss: 0.214620
Train Epoch: 15 [32000/50000 (64%)]	Loss: 0.337319
Train Epoch: 15 [38400/50000 (77%)]	Loss: 0.186454
Train Epoch: 15 [44800/50000 (90%)]	Loss: 0.299778
Train Epoch: 16 [0/50000 (0%)]	Loss: 0.226033
Train Epoch: 16 [6400/50000 (13%)]	Loss: 0.139336
Train Epoch: 16 [12800/50000 (26%)]	Loss: 0.207583
Train Epoch: 16 [19200/50000 (38%)]	Loss: 0.208954
Train Epoch: 16 [25600/50000 (51%)]	Loss: 0.219778
Train Epoch: 16 [32000/50000 (64%)]	Loss: 0.219288
Train Epoch: 16 [38400/50000 (77%)]	Loss: 0.219795
Train Epoch: 16 [44800/50000 (90%)]	Loss: 0.313411
Train Epoch: 17 [0/50000 (0%)]	Loss: 0.280990
Train Epoch: 17 [6400/50000 (13%)]	Loss: 0.184363
Train Epoch: 17 [12800/50000 (26%)]	Loss: 0.183690
Train Epoch: 17 [19200/50000 (38%)]	Loss: 0.117576
Train Epoch: 17 [25600/50000 (51%)]	Loss: 0.198941
Train Epoch: 17 [32000/50000 (64%)]	Loss: 0.265863
Train Epoch: 17 [38400/50000 (77%)]	Loss: 0.234642
Train Epoch: 17 [44800/50000 (90%)]	Loss: 0.136114
Train Epoch: 18 [0/50000 (0%)]	Loss: 0.296832
Train Epoch: 18 [6400/50000 (13%)]	Loss: 0.163227
Train Epoch: 18 [12800/50000 (26%)]	Loss: 0.106059
Train Epoch: 18 [19200/50000 (38%)]	Loss: 0.122336
Train Epoch: 18 [25600/50000 (51%)]	Loss: 0.077063
Train Epoch: 18 [32000/50000 (64%)]	Loss: 0.108953
Train Epoch: 18 [38400/50000 (77%)]	Loss: 0.090041
Train Epoch: 18 [44800/50000 (90%)]	Loss: 0.152034
Train Epoch: 19 [0/50000 (0%)]	Loss: 0.120802
Train Epoch: 19 [6400/50000 (13%)]	Loss: 0.033152
Train Epoch: 19 [12800/50000 (26%)]	Loss: 0.089820
Train Epoch: 19 [19200/50000 (38%)]	Loss: 0.054755
Train Epoch: 19 [25600/50000 (51%)]	Loss: 0.055792
Train Epoch: 19 [32000/50000 (64%)]	Loss: 0.100382
Train Epoch: 19 [38400/50000 (77%)]	Loss: 0.098241
Train Epoch: 19 [44800/50000 (90%)]	Loss: 0.100497
Train Epoch: 20 [0/50000 (0%)]	Loss: 0.119685
Train Epoch: 20 [6400/50000 (13%)]	Loss: 0.065726
Train Epoch: 20 [12800/50000 (26%)]	Loss: 0.076239
Train Epoch: 20 [19200/50000 (38%)]	Loss: 0.024349
Train Epoch: 20 [25600/50000 (51%)]	Loss: 0.051095
Train Epoch: 20 [32000/50000 (64%)]	Loss: 0.046113
Train Epoch: 20 [38400/50000 (77%)]	Loss: 0.050620
Train Epoch: 20 [44800/50000 (90%)]	Loss: 0.029069
```
![filter4_result](./output/filter4_result.png?raw=true)


 ## Authors:
  - [Husam Lucia](https://www.linkedin.com/in/husam-lucia-6841b51a3)
  - [Regev Aloni](https://github.com/AloniRegev)
