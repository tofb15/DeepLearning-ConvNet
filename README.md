# DeepLearning-ConvNet
Deep Convolutional Neural Network implemented using CUDA and SFML

This project is my first atempt to implement a Deep Neural Network.
* ( Project started                     @ 5:th dec 2018 )
* ( Targeted deadline for the main goal @ 5:th jan 2019 )

-==The Goal==-

-MAIN-
The program is that it should be able to classifi a set of 60000 images(of size 32x32x3 each) to one of 10 different classes.
The program should utilze the GPU using CUDA.

-Bonus-
The program should still be generic enough to solve any kind of problem where deep learning can be applied.

-==Short Description==-

* Visual Studio Community 2017, version 15.9.1
* SFML-2.5.1 is included in the project and is used to display the images to the screen.
* Program is using CUDA 10.0 which must be installed on your computer and integrated with Visual Studio 2017. (https://developer.nvidia.com/cuda-downloads)

Image dataset included: CIFAR-10. Images are saved in .bin files, each file containing 10000 images, of size 32x32x3 each, and the class each image belong to. Description of the dataset ->(http://www.cs.utoronto.ca/~kriz/cifar.html)

