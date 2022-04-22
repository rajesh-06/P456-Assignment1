import math as m
import matplotlib.pyplot as plt
import mm1


def matrix_function(x, y, n):
    i1 = x%n
    i2 = y%n
    j1 = x//n
    j2 = y//n
    if x == y:
        return -0.96
    if ((i1+1)%n,j1) == (i2,j2):
        return 0.5
    if (i1,(j1+1)%n) == (i2,j2):
        return 0.5
    if ((i1-1)%n,j1) == (i2,j2):
        return 0.5
    if (i1,(j1-1)%n) == (i2,j2):
        return 0.5
    
    return 0



n = 20
n2 = n**2
eps = 1e-6

I=mm1.identity_mat(n2)

#calculating only 1st 2 columns of the inverse matrix
for j in range(2):
    A1 = [[I[i][j]] for i in range(n2)]
    #print("A",A1)
    A1,it,res=mm1.conjugate_gradient_on_the_fly(matrix_function, A1, eps)             # sending the function as argument instead of matrix                                                                                 
    print("column=",j)
    for i in range(n2):
        I[i][j] = A1[i][0]

#writing 1st 2 columns in q3_output.txt file
f = open("q3_output.txt", "w+")
for i in range(n2):
	f.write(str(round(I[i][0],6))+"	"+str(round(I[i][1],6))+"\n")

f.close()
plt.plot(it,res,"o-",label='conjugate gradient method')
plt.xlabel('Iterations')
plt.ylabel('Residue')
plt.yscale('log')
plt.legend()
plt.savefig('q3_fig.png')
plt.show()