# 3D-ConvNeuralNet-material-property-prediction

# Reference paper
This repo includes the dataset/scripts of 3D Convolutional Neural Network for material property prediction in paper: 

[Rao, Chengping, and Yang Liu. "Three-dimensional convolutional neural network (3D-CNN) for heterogeneous material homogenization." Computational Materials Science 184 (2020): 109850.](https://www.sciencedirect.com/science/article/abs/pii/S0927025620303414)

# Description for each file
- **3D_CNN_training**: Training scripts ;
- **data**: The dataset for this application contains 2000 samples or (**X, Y**) pairs. **X** is the tensor of size 101x101x101 that carries the phase indication for a two-phase heterogeneous material microstructure (or Representative Volume Element). **Y** is a vector of size 12 that represents the effective material properties for the microstructure.
- **saved_model**: The model with best performance on validation set.
- **gallery**: Some figures.

# Overview

<img src="https://github.com/Raocp/3D-ConvNeuralNet-material-property-prediction/blob/master/gallery/clous_pt.png?raw=true" width="500">

> Input **X** for the 3D Convolutional Neural Net, i.e. Cartisian grid carrying phase indication

![](https://github.com/Raocp/3D-ConvNeuralNet-material-property-prediction/blob/master/gallery/NN_archi.png?raw=true)

> Architecture of the 3D Convolutional Neural Net



# Note
- The code was developed based on TensorFlow 1.10.0 and Keras 2.2.4. All the runtime performance are evaluated on GeForce GTX 1080 Ti.
