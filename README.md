# Visualising the operation of SOMs

## U-Matrix Visualistion of the training process of Self-Organising Maps

### There are two programs in this project:

**Basic Self-Organising Map with U-Matrix implementation (U_SOM.py):**

This program creates a Self-Organising Map based on user input parameters for the map dimensions, learning rate and number of training epochs. The user also chooses the input dataset on which to train the SOM. The SOM is trained on the input dataset, and the values for a U-Matrix visualisation of the trained SOM are calculated at the end of each epoch and saved to array files. At the end of the training process, a 3D U-Matrix Visualisation program is run, reproducing the calculated U-Matrix for the SOM.

SOM works on numerical datasets only, which include a class for each data sample. The class does not have to be numerical.

Input datasets must be in .csv format, with each row a new data sample, and the final column containing the class of each sample

**3D U-Matrix Visualisation Program in PyOpenGL (U_Matrix_Visualisation.py):**

This program creates a 3D U-Matrix Visualisation in PyOpenGL of the training process of a SOM, using SOM training data and U-Matrix calculations provided by the U_SOM.py program. The allowes the user to view and switch between the 3D U-Matrix of each training epoch of the U_SOM.py program

### Instructions:

Download all the files in this repository and place into a single project folder.

To run these programs, the user must first ensure that Python 3.7 is installed (and pip, which should come with Python 3.7) 

With Python 3.7 installed, open a command window and type the following commands to install the Numpy and PIL libraries:
	
	pip install Numpy
	pip install Pillow

The PyOpenGL and GLUT libraries are also required. The standard distribution of PyOpenGL comes with GLUT, but it is broken.
To install the correct version, navigate to the Installation folder (in the project files) in the command line. Inside are two PyOpenGL packaged distributions, one for 32-bit Python and one for 64-bit Python. _NOTE: A 64-bit Windows version does not mean that you have 64-bit Python_.
To install these packages, make sure you are in the correct directory in command line and type:

	pip install PyOpenGL-3.1.3b2-cp37-cp37m-win_amd64.whl (for 64-bit Python)
OR

	pip install PyOpenGL-3.1.3b2-cp37-cp37m-win32.whl (for 32-bit Python)

Once installed, navigate back to the main project folder in the command line.

If Python has been added to the system path, run the program by:

	python U_SOM.py

If Python is not added to the system path, the user will have to find the Python installation directory and then run:

	user_python_installation_directory/python.exe U_SOM.py

The user_python_installation_directory should usually look something like:

	C:\Users\USERNAME\AppData\Local\Programs\Python\Python37
	
Where USERNAME is replaced by your personal computer username.

Once the program is running, follow the program instructions in the command line.
