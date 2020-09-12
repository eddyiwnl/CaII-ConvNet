# CaII-ConvNet
The purpose of this project is to use machine learning to discover CaIIλλ3934, 3969 absorption spectra from Sloan Digital Sky Survey (SDSS) Data Release 14. Ultimately, the 
goal of studying absorption lines from the spectra of quasars is so sufficient data can be provided to answer questions about some very fundamental problems in cosmology and 
astronomy. For example, questions about the formation and evolution of galaxies, and about the cosmic evolution of the UV ionizing background.

## Methodology
#### 1. Preprocessing
I downloaded all the spectra plate files from the SDSS DR14, at https://www.sdss.org/dr14/data_access/bulk/, and https://www.sdss.org/dr14/algorithms/qso_catalog/. I then apply 
a redshift filter and emissions, check for strong absorption lines, and obtain my dataset for training. In addition to this dataset, because there are too few discovered 
CaII absorbers, a neural network cannot be trained successfully. Thus, I have generated 69584 artificial spectra, with a portion of them having a strong CaII profile.

#### 2. Convolutional Neural Network Structure
My neural network has 5 hidden layers, 3 pooling layers, and 5 stages. The input layer is of size 3841, because that is the dimension of the data I am feeding into the 
ConvNet. My kernel sizes for each layer are 15, 5, 3, 3, and 3, respectively. Simplified, the architecture was as follows: a convolutional layer, a ReLU layer with sigmoid and tanh 
functions, a pooling layer, and a fully-connected layer. 

## Contributors
Thanks to Yinan Zhao and Professor Jian Ge for assistance in creating the neural network and generating artificial spectra.
