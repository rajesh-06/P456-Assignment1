import mm1
import copy
with open('q1_mat.txt', 'r') as f:
    A = [[float(num) for num in line.split()] for line in f]
print("Matrix A=")
for i in A:
    print(i)
A1=copy.deepcopy(A)
A2=copy.deepcopy(A)
with open('q1_mat2.txt', 'r') as f:
    B = [[float(num) for num in line.split()] for line in f]
B=mm1.transpose(B)[0]
B1=copy.deepcopy(B)
print("Matrix B=\n",B)
print("")


print("Solving using Gauss-Jordan elimination")
mm1.gauss_jordan(A2,B1)
print('Matrix X =\n',B1)
#checking for result A*X==B
print("A*X =\n",mm1.mat_vec_mult(A1,B1))
print("")


print('\nSolving using LU decomposition method:')
#LU decompose of matrix A
A,B=mm1.lu_decompose(A,B)

#solving for solution matrix X
x=mm1.lux(A,B)
print('Matrix X =\n',x)

#checking for result A*X==B
print("LUx =\n",mm1.mat_vec_mult(A1,x))


#------------------------------output----------------------------------
'''
Matrix A=
[1.0, -1.0, 4.0, 0.0, 2.0, 9.0]
[0.0, 5.0, -2.0, 7.0, 8.0, 4.0]
[1.0, 0.0, 5.0, 7.0, 3.0, -2.0]
[6.0, -1.0, 2.0, 3.0, 0.0, 8.0]
[-4.0, 2.0, 0.0, 5.0, -5.0, 3.0]
[7.0, 0.0, -1.0, 5.0, 4.0, -2.0]
Matrix B=
 [19.0, 2.0, 13.0, -7.0, -9.0, 2.0]

Solving using Gauss-Jordan elimination
Matrix X =
 [-9.677580355360385, -27.20206960803895, -9.01966460371336, 7.229753111066779, 7.4070007320157165, 2.5266852997937144]
A*X =
 [18.999999999999986, 2.00000000000003, 12.999999999999993, -7.000000000000032, -8.9999999999999, 1.9999999999999991]


Solving using LU decomposition method:
Matrix X =
 [-9.67758035536037, -27.20206960803894, -9.019664603713348, 7.229753111066765, 7.407000732015724, 2.5266852997937086]
LUx =
 [19.0, 2.000000000000007, 12.999999999999996, -7.000000000000007, -9.000000000000068, 2.0000000000000604]
 '''