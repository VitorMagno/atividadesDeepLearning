# %%
# Numpy Array
import numpy as np
# %% 
# creating an empty and full np.array
fullArray = np.array([1,2,3,4])
emptyArray = np.empty(4)
print(fullArray)
print(emptyArray)

# %%
# array filled with zero
zeros = np.zeros(4)
print(zeros)

# %%
# filled with one
one = np.ones(4)
print(one)

# %%
# checking for a specific row
matrix = np.array([[1,2,3,4], [5,6,7,8]])
print(matrix[1,:])

# %%
# removing rows containing non-numeric values
nonnumeric = np.array([[14,np.nan,np.nan],[1,2,3]])
nonnumeric = nonnumeric[~np.isnan(nonnumeric).any(axis=1)]
nonnumeric

# %%
# removing single dimensional entries
nonnumeric = nonnumeric[0,:]
print(nonnumeric)

# %%
# finding sequences
a = np.array([1234, 1542, 1234, 1567])
count = (a==1234).sum()
print(count)

# %%
# most frequent value in a np.array
arr = np.array([2, 3, 4, 5, 3, 4, 5, 3, 5, 4, 7, 8, 3, 6, 2])

def mostFrequent(arr):
    value, count = np.unique(arr, return_counts=True)
    frequentIndex = np.argmax(count)
    return value[frequentIndex]

result = mostFrequent(arr)
print(result)

# %%
# combining one and two dimensional arrays
arr1 = np.array([1., 2., 3., 4.,])
arr2 = np.array([5.,6.,7.,8.])
arr4 = np.array([[0,6],[9,1]])
arr5 = np.array([[8,4],[6,6]])
arr6 = np.column_stack((arr4,arr5))
arr3 = np.column_stack((arr1,arr2))
print(arr3 ,arr3.shape,'\n', arr6, arr6.shape)

# %%
# creating an array of all combinations(2,2) of another two arrays
result = np.meshgrid(arr1,arr2)
result = np.array(result).T.reshape(-1,2)
print(result)

# %%
# making a border around an array
modes = ['edge','linear_ramp','maximum','mean','median','minimum','reflect','symmetric', 'wrap','empty','constant']
for i in modes:
    arrBorder = np.pad(arr3, pad_width=1, mode=i)
    print(i)
    print(arrBorder)

# %%
# comparing two numpy arrays
result = np.array_equal(arr1,arr2)
print(result)

# %%
# checking for a specified value
value = 1
result = np.isin(arr1, value) # np.array 
result2 = (arr1 == 1) 
print(result, result2)

# %%
# get all 2d diagonals of a 3d numppy array
array = np.array([[[1, 2, 3], [4, 5, 6],[7, 8, 9]],[[10, 11, 12],[13, 14, 15],[16, 17, 18]],[[19, 20, 21],[22, 23, 24],[25, 26, 27]]])
diagonals = []
for i in range(array.shape[0]):
    diagonals.append(np.diagonal(array[i]))
diagonals = np.array(diagonals)
print(diagonals)

# %%
# flattening a matrix 
print(array[0,:])
flatten = array[0,:].flatten()
print(flatten)

# %%
# flattening a 2d numpyarray
print(arr2)
print(arr2.flatten())

# %%
# moving axes 
print(array)
moved = np.moveaxis(array, [0,1], [1,0])
print(moved)

# %%
# interchanging axes
print(array)
changed = np.swapaxes(array, 0,1)
print('\n')
print(changed)

# %%
# fibonacci series
x = 10
x = np.arange(1, x)
sqrtFive = np.sqrt(5)
alfa = (1 + sqrtFive) / 2
beta = (1 - sqrtFive) / 2

Fn = np.rint(((alfa**x) - (beta**x)) / sqrtFive)
Fn

# %%
# counting number of non-zero values
randomArray = np.rint(np.random.rand(10)*10)
# %%
print(randomArray)
resp = 0
for element in randomArray:
    if(element>0):
        resp+=1
resp

# %%
# counting the number of elements along an axis
len(array[0])

# %%
# triming the leading and/or trailing zeros from a 1-D array
toTrim = np.array([0,0,1,3,4,0])
toTrim = np.trim_zeros(toTrim)
print(toTrim)

# %%
# changing data type
print(toTrim.dtype)
toTrim = toTrim.astype('float64')
print(toTrim)
print(toTrim.dtype)

# %%
# reversing a numpy array
toTrim = np.flip(toTrim)
print(toTrim)

# %%
# making an array readOnly
toTrim.setflags(write=False)
toTrim[1]=2

# %%
# Numpy Matrix
matrix = np.rint(np.random.rand(3,3)*10)
matrix
# %%
# Getting the maximum value of a matrix
matrix.max()

# %%
# Getting the minimum value of a matrix
matrix.min()

# %%
# shape of a matrix
matrix.shape

# %%
# selecting elements from a given matrix
print(matrix[0][1])
print(matrix[:1])

# %%
# the sum of values
matrix.sum()

# %%
# sum of diagonal
soma = np.trace(matrix)
print(soma)

# %%
# adding and subtracting
matrix2 = np.rint(np.random.rand(3,3)*10)
addM = matrix + matrix2
print(addM)
subM = matrix - matrix2
print(subM)

# %%
# adding columns and rows
column = np.array([1,2,3])
row = np.array([1,2,3,4])
novoArray = np.hstack((matrix, np.atleast_2d(column).T))
print(novoArray)
novoArray = np.vstack((novoArray, row))
print('\n',novoArray)

# %%
# multiplication of a matrix
multM = np.dot(matrix, matrix2)
print(multM)

# %%
# eigenvalues
eigenvalues, eigvectors = np.linalg.eig(matrix)
print(eigenvalues)

# %%
# determinant
determinant = np.linalg.det(matrix)
print(determinant)

# %%
# inverting a matrix
inverseMatrix = np.linalg.inv(matrix)
print(inverseMatrix)

# %%
# counting unique values
print(len(np.unique(matrix)))

# %%
# multiplying matrix of complex numbers
x = np.array([2+3j, 4+5j])
print(x)
y = np.array([8+7j, 5+6j])
print(y)
z = np.vdot(x, y)
print(z)

# %%
# outer product
z = np.outer(x,y)
print(z)

# %%
# inner, outer, cross products
z = np.outer(x,y)
print(z)
z = np.inner(x,y)
print(z)
z = np.cross(x,y)
print(z)

# %%
# covariance
x = np.array([[0,1,2],[2,1,0]])
z = np.cov(x)
print(z)

# %%
# 