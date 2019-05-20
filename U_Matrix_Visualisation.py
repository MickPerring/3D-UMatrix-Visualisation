'''
* Full U_Matric_Visualisation.py program

* Author: Mick Perring
* Date: May 2019

* This program creates a 3D U-Matrix Visualisation in PyOpenGL of
* the training process of a SOM, using SOM training data and
* U-Matrix calculations provided by the U_SOM.py program. The
* allowes the user to view and switch between the 3D U-Matrix of
* each training epoch of the U_SOM.py program
'''

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import numpy as np

class U_MATRIX(object):

    def __init__(self, dims, epchs):

        self.dim = dims
        self.scale = int(dims/4)
        self.ld = int(-(dims-1)/2)
        self.rd = int(((dims-1)/2)+1)
        self.epochs = epchs
        self.epoch = 1
        self.r = 0.0
        self.g = 0.0
        self.b = 0.0
        self.rr = 0.1
        self.gg = 0.1
        self.bb = 0.1
        self.R = False
        self.G = False
        self.B = False

    '''
    * This method handles window resizing and reshaping
    '''
    def resize(self, w, h):

        if h == 0:
            h = 1
        ratio = 1.0*w/h

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glViewport(0, 0, w, h)
        gluPerspective(45,ratio,1,1000)
        glMatrixMode(GL_MODELVIEW)

    '''
    * This method handles keyboard input
    '''
    def keyboard(self, bkey, x, y):

        key = bkey.decode("utf-8")

        if key == chr(102):  # F
            glutFullScreenToggle()
        if key == chr(101):  # E
            if self.epoch < self.epochs:
                self.epoch += 1
        if key == chr(113):  # Q
            if self.epoch > 1:
                self.epoch -= 1
        if key == chr(114):  # R
            if not self.R:
                self.r = 0.9
                self.rr = 1.0
                self.g = 0.0
                self.gg = 0.1
                self.b = 0.0
                self.bb = 0.1
                self.R = True
                self.G = False
                self.B = False
            else:
                self.r = 0.0
                self.g = 0.0
                self.b = 0.0
                self.rr = 0.1
                self.gg = 0.1
                self.bb = 0.1
                self.R = False
        if key == chr(103):  # G
            if not self.G:
                self.r = 0.0
                self.rr = 0.1
                self.g = 0.9
                self.gg = 1.0
                self.b = 0.0
                self.bb = 0.1
                self.R = False
                self.G = True
                self.B = False
            else:
                self.r = 0.0
                self.g = 0.0
                self.b = 0.0
                self.rr = 0.1
                self.gg = 0.1
                self.bb = 0.1
                self.G = False
        if key == chr(98):  # B
            if not self.B:
                self.r = 0.0
                self.rr = 0.1
                self.g = 0.0
                self.gg = 0.1
                self.b = 0.9
                self.bb = 1.0
                self.R = False
                self.G = False
                self.B = True
            else:
                self.r = 0.0
                self.g = 0.0
                self.b = 0.0
                self.rr = 0.1
                self.gg = 0.1
                self.bb = 0.1
                self.B = False
        if key == chr(27):  # ESC
            sys.exit(0)

    '''
    * This method print all HUD information
    '''
    def text(self, x, y, r, g, b, text):

        glColor3f(r, g, b)

        glRasterPos2f(x, y)

        for i in range(len(text)):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(text[i]))

    '''
    * This method sets the 3D environment lighting
    '''
    def light(self):
        glLightfv(GL_LIGHT0, GL_AMBIENT, GLfloat_4(0.0, 0.0, 0.0, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, GLfloat_4(1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, GLfloat_4(1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(1.0, 1.0, 1.0, 0.0))
        glLightfv(GL_LIGHT1, GL_AMBIENT, GLfloat_4(-0.5, -0.5, -0.5, 0.1))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, GLfloat_4(1.0, 1.0, 1.0, 0.1))
        glLightfv(GL_LIGHT1, GL_SPECULAR, GLfloat_4(1.0, 1.0, 1.0, 0.1))
        glLightfv(GL_LIGHT1, GL_POSITION, GLfloat_4(-2.0, 2.0, 2.0, 0.0))
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)

    '''
    * This method sets the 3D object materials
    '''
    def material(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, GLfloat_4(0.8, 0.8, 0.8, 1.0))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, GLfloat_4(0.8, 0.8, 0.8, 1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR, GLfloat_4(1.0, 1.0, 1.0, 1.0))
        glMaterialfv(GL_FRONT, GL_SHININESS, GLfloat(10.0))

    '''
    * This method cis the main rendering function that renders everything in the 3D environment
    '''
    def render(self):

        glClearColor(0.1, 0.1, 0.1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        gluLookAt(
            0, int(1.5*self.dim), self.dim,
            0, 0, 0,
            0.0, 1.0, 0.0
        )

        self.light()
        self.material()
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)

        glColor3f(0.9, 0.9, 0.9)

        ldim = self.ld-0.5
        rdim = self.rd-0.5

        glBegin(GL_QUADS)
        glVertex3f(ldim, 0.0, ldim)
        glVertex3f(ldim, 0.0, rdim)
        glVertex3f(rdim, 0.0, rdim)
        glVertex3f(rdim, 0.0, ldim)
        glEnd()

        self.nodes()

        x, y, width, height = glGetDoublev(GL_VIEWPORT)
        x_left = x+10
        x_right = x+width-10
        y_bottom = y+20
        y_top = y+height-30

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0.0, width, 0.0, height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        epoch_str = "Epoch: {0}".format(self.epoch)
        str_len = glutBitmapWidth(GLUT_BITMAP_HELVETICA_18, ord("W"))*len(epoch_str)/2

        f = open("Temp\\data_{0}.txt".format(self.epoch), 'r')
        data = f.read()

        self.text(x_right-str_len, y_top, 0.0, 1.0, 0.0, epoch_str)
        self.text(x_left, y_top, 0.0, 1.0, 0.0, "Keys:")
        self.text(x_left, y_top-25, 0.0, 1.0, 0.0, "Q - Previous Epoch")
        self.text(x_left, y_top-50, 0.0, 1.0, 0.0, "E - Next Epoch")
        self.text(x_left, y_top-75, 0.0, 1.0, 0.0, "F - Toggle Fullscreen")
        self.text(x_left, y_top-100, 0.0, 1.0, 0.0, "ESC - Exit Program")
        self.text(x_left, y_bottom, 0.0, 1.0, 0.0, data)

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glEnable(GL_TEXTURE_2D)

        glutSwapBuffers()

    '''
    * This method creates all of the 3D U-Matrix objects to be rendered
    '''
    def nodes(self):

        name = 'Temp\\array_{0}.npz'.format(self.epoch)

        npz_array = np.load(name)

        colour_arr = npz_array[npz_array.files[0]]

        colour_arr = colour_arr/255.0

        for i in range(self.ld, self.rd):
            for j in range(self.ld, self.rd):
                colour = colour_arr[i-self.ld][j-self.ld]
                glColor3f(colour+self.r, colour+self.g, colour+self.b)

                blocks = int(self.scale*(1.0-colour))

                glPushMatrix()
                glTranslatef(j, 0.0, i)

                glutSolidCube(1.0)

                for k in range(blocks):
                    glTranslate(0.0, 1.0, 0.0)
                    glColor3f(colour+self.r, colour+self.g, colour+self.b)
                    glutSolidCube(1.0)
                    glColor3f(colour+self.rr, colour+self.gg, colour+self.bb)
                    glutWireCube(1.0)

                glPopMatrix()

'''
* This is the main method of the program. It initialises the OpenGL window and calls all
* of the main functions in the main loop
'''
def main():

    glutInit()
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(560, 200)
    glutCreateWindow("3D U-Matrix Visualisation in PyOpenGL")
    glutFullScreen()

    file = np.load('Temp\\data.npz')

    data = file[file.files[0]]

    uMat = U_MATRIX(data[0], data[1])

    glutDisplayFunc(uMat.render)
    glutReshapeFunc(uMat.resize)
    glutIdleFunc(uMat.render)

    glutKeyboardFunc(uMat.keyboard)

    glEnable(GL_DEPTH_TEST)

    glutMainLoop()


if __name__ == "__main__":
    main()
