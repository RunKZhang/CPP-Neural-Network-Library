# CPP-Neural-Network-Library-From-Scratch
This is a header-only cpp neural network library from scratch and it is for my undergraduate thesis.

# Prerequisites
CMake 3.23.0: https://cmake.org/download/

This project is built by using CMake, so it is necessary to install CMake in your computer before compile the project.

Eigen 3.4.0 or above: https://eigen.tuxfamily.org/index.php?title=Main_Page

autodiff 0.6.7 or above: https://github.com/autodiff/autodiff

If you have installed the two libraries mentioned above, please put the folders of them under the root directory of this project.

# FIles explanation 
*src* has the directory of all the header files and main program contained.

*neurobase.h* contains the base class for the neurons of network.

*mse.h* and *fully_connected.h* are inherited from the *neurobase.h*

*activ_func.h* contains some often used activation functions.

*process_data.h* is used to read datasets in .csv format.

*test_nn.h* is used to test regression of functions.

"exemain.cpp" is the entrance of the project.

# Run
To run a scipt, I add a example dataset called: Iris.csv to the repository. The user can use it to begin.
