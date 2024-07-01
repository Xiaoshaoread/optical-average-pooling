# Environmental dependence

The **requirements.txt** specifies the minimum configuration of the library required for the code to run

# File description

**torchonn**: This is the basic code for configuring ONN convolution and matrix dot product operations. The main() function should not be present here.

**exsitu_training:** This file contains the code for the offline training section. It is mainly divided into two parts. First, you need to train an OCNN model and save the model parameter file. 2, call the previous model for testing, and can introduce the phase noise of MZI.

**on-chip__training**: This file contains the simulation portion of on-chip training. It mainly involves introducing noise in the training process and indirectly affecting the updating of weight parameters by changing the phase.

# tips

The files in the code repository should remain in their original order to avoid partial library calls failing.

