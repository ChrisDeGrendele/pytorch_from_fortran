# A pytorch NN called from fortran.

I couldn't find a MWE of calling a neural network defined in python from fortran. Here's calling a resnet from fortran.

If you're here. *I'm sorry.* Hopefully it's as simple as copying and pasting this code into your project :)


We bridge Fortran and Python using a C helper library (built with Python/NumPy C APIs), allowing the Fortran code (via iso_c_binding) to directly execute the PyTorch model and exchange numerical data.
