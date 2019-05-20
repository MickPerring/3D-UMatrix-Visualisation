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

To run these programs, the user must first ensure that Python 3.7 and OpenGL are installed on the Windows system, as well as the GLUT/FreeGLUT library.

Once installed, navigate to this project folder in the command line.

If Python has been added to the system path, run the program by:

	python U_SOM.py

If Python is not added to the system path, the user will have to find the Python installation directory and then run

	user_python_installation_directory/python.exe U_SOM.py

Once program is running, follow the instructions in the command line
