#using jacobi and guass-seidel to find the inverse of the given matrix

import mm1
import copy
import numpy as np
import matplotlib.pyplot as plt

b=[-5/3,2/3,3.0,-4/3,-1/3,5/3]
with open('q2.txt','r') as f:
    A = [[float(num) for num in line.split()] for line in f]
A1=copy.deepcopy(A)
b1=copy.deepcopy(b)

print('Matrix A =')
for i in A:
    print(i)
print('')

#Solving using Jacobi method:
print('Solving for inverse of matrix A using Jacobi method:')
I=mm1.identity_mat(len(A))
for i in range(len(A)):
	A2=copy.deepcopy(A)
	I[i],error,residue_j = mm1.jacobi(A2,I[i],1.0e-4)

#Inverse of A
I=mm1.transpose(I)
print('Matrix A_inverse =')
for i in I:
    print(i)
print('')

#Checking for A*A_inverse == I
AA_inv=mm1.mat_mult(A,I)

for i in range(len(A)):
	for j in range(len(A)):
		AA_inv[i][j]=round(AA_inv[i][j],3)#rounding off the values to 3 decimal places
print('A*A_inverse =')
for i in AA_inv:
    print(i)
print('')

#Solving using Gauss-Seidel method:
print('Solving for inverse of matrix A using Gauss-Seidel method:')
I=mm1.identity_mat(len(A))
A_inv, residue_gs = mm1.gauss_seidel_inverse(A,1.0e-4)

print('A_inverse = ')
for i in A_inv:
    print(i)
print('')

#Checking for A*A_inverse == I
AA_inv=mm1.mat_mult(A,A_inv)

for i in range(len(A)):
	for j in range(len(A)):
		AA_inv[i][j]=round(AA_inv[i][j],3)#rounding off the values to 3 decimal places

print('A*A_inverse =')
for i in AA_inv:
    print(i)
print('')

plt.plot(residue_j[0],residue_j[1],"-o",label='Jacobi')
plt.plot(residue_gs[0],residue_gs[1],"-*",label='Gauss-Seidel')
plt.xlabel('Iteration Step')
plt.ylabel('Residue')
plt.legend()
plt.grid()
plt.show()


'''
Matrix A =
[2.0, -3.0, 0.0, 0.0, 0.0, 0.0]
[-1.0, 4.0, -1.0, 0.0, -1.0, 0.0]
[0.0, -1.0, 4.0, 0.0, 0.0, -1.0]
[0.0, 0.0, 0.0, 2.0, -3.0, 0.0]
[0.0, -1.0, 0.0, -1.0, 4.0, -1.0]
[0.0, 0.0, -1.0, 0.0, -1.0, 4.0]

Solving for inverse of matrix A using Jacobi method:
Matrix A_inverse =
[0.9351800634741533, 0.8702457873409919, 0.2598643086904607, 0.2079002613636324, 0.41568618311995026, 0.16895012413476573]
[0.2901022345786059, 0.5801420955130241, 0.1732250647228108, 0.13859094247346287, 0.27711951130273815, 0.1126217827628551]
[0.08660771611469088, 0.17318799199845644, 0.3203760907112998, 0.05630298792945926, 0.11257853562799319, 0.10825365617168992]
[0.2079002613636324, 0.41568618311995026, 0.16895012413476573, 0.9351800634741533, 0.8702457873409918, 0.2598643086904606]
[0.1385909424734629, 0.2771195113027382, 0.11262178276285513, 0.2901022345786059, 0.5801420955130241, 0.1732250647228108]
[0.05630298792945926, 0.11257853562799319, 0.10825365617168992, 0.08660771611469088, 0.17318799199845641, 0.3203760907112998]

A*A_inverse =
[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

Solving for inverse of matrix A using Gauss-Seidel method:
A_inverse =
[0.9351223158891565, 0.8701776888079604, 0.25979692074060245, 0.20784746492905842, 0.41564948983675304, 0.168885546276172]
[0.29006890342296876, 0.5801079251609023, 0.1731854652282891, 0.13855280394633862, 0.27708532459184265, 0.11257838529964545]
[0.0865893069684121, 0.17316785702995363, 0.3203554250678569, 0.056285935415103155, 0.11256456918112426, 0.10823384600756197]
[0.20783916069686637, 0.4156235440888213, 0.16887753272682665, 0.9351101502237152, 0.8701831183222202, 0.2597847550751611]
[0.1385490971426287, 0.2770737430521589, 0.11257480824956356, 0.29006347297103197, 0.5801103487616793, 0.17318003477635224]
[0.0562846010277602, 0.11256040002052814, 0.10823255832935512, 0.08658735209653379, 0.1731687294857009, 0.32035347019597854]

A*A_inverse =
[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]'''