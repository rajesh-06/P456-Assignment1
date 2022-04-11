#This code solve for Ax = b for the question 2 using LU and Jacobi.

import mm1
import copy
import numpy as np
import matplotlib.pyplot as plt

b=[-5/3,2/3,3.0,-4/3,-1/3,5/3]
with open('q2.txt','r') as f:
    A = [[float(num) for num in line.split()] for line in f]
A1=copy.deepcopy(A)
A2=copy.deepcopy(A)
b1=copy.deepcopy(b)
print('Matrix A =')
for i in A:
    print(i)
print('Matrix b =\n',b)

#Solving using LU decomposition method:
print('Solving using LU deomposition method:')
#LU decompose of matrix A
A,B=mm1.lu_decompose(A,b)
#solving for solution matrix X
x=mm1.lux(A,B)
print('Matrix X =\n',x)
#checking for result A*X==B
AX=mm1.mat_vec_mult(A1,x)
print('A*X =\n',AX)
print('\n')

#Solving using Jacobi method:
print('Solving using Jacobi method:')
#solving for solution matrix X
x,error_jacobi,residue=mm1.jacobi(A1,b1,1.0e-4)
print('Matrix X =\n',x)
#checking for result A*X==B
ax=mm1.mat_vec_mult(A2,x)
print('A*X =\n',ax)

#-----------------------output-------------------
'''Matrix A =
[2.0, -3.0, 0.0, 0.0, 0.0, 0.0]
[-1.0, 4.0, -1.0, 0.0, -1.0, 0.0]
[0.0, -1.0, 4.0, 0.0, 0.0, -1.0]
[0.0, 0.0, 0.0, 2.0, -3.0, 0.0]
[0.0, -1.0, 0.0, -1.0, 4.0, -1.0]
[0.0, 0.0, -1.0, 0.0, -1.0, 4.0]
Matrix b =
 [-1.6666666666666667, 0.6666666666666666, 3.0, -1.3333333333333333, -0.3333333333333333, 1.6666666666666667]   
Solving using LU deomposition method:
Matrix X =
 [-0.33333333333333337, 0.3333333333333333, 1.0, -0.6666666666666665, 5.401084984662924e-17, 0.6666666666666667]A*X =
 [-1.6666666666666667, 0.6666666666666665, 3.0, -1.3333333333333333, -0.3333333333333333, 1.666666666666667]    


Solving using Jacobi method:
Matrix X =
 [-0.3332219651295557, 0.33339798371449714, 1.0000267271272278, -0.6665481534824041, 6.075270763257479e-05, 0.6666951085095364]
A*X =
 [-1.6666378814026028, 0.6667264201526839, 3.000013816284878, -1.333278565087706, -0.3333019279110991, 1.6666929542032851]'''