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
# converting covariance to correlation using python
y = np.array([[4,3,0],[0,3,4]])
corr = np.cov(x,y)/(np.std(x)*np.std(y))
print(corr)

# %%
# kronecker product
kron = np.kron(x,y)
print(kron)

# %%
# Convert the matrix into a list
listaX = x.tolist()
print(listaX)

# %%
# Numpy indexing 

# %%
# Replace NumPy array elements that doesn’t satisfy the given condition
x[x>0] = 1
print(x)

# %%
# Return the indices of elements where the given condition is satisfied
a = np.where(x>0)
print(a)

# %%
# Replace nan values
x = np.array([np.nan, 1, np.nan])
x = np.nan_to_num(x, nan=-1)
x

# %%
# Replace negative value with zero in numpy array
x[x<0] = 0
x

# %%
# How to get values of an NumPy array at certain index positions?
element = y[0][1]
print(element)

# %%
# Find indices of elements equal to zero in a NumPy array
ind = np.where(x == 0)
print(ind)

# %%
# How to Remove columns in Numpy array that contains non-numeric values?
y = np.array([[10.5, 22.5, 3.8],[23.45, 50, 78.7],[41, np.nan, np.nan]])
y = y[:,~np.isnan(y).any(axis=0)]
y

# %%
# How to access different rows of a multidimensional NumPy array?
y[[0,2]]

# %%
# Get row numbers of NumPy array having element larger than X
x = np.array([[1,3,2],[5,6,7]])
row = np.where(np.any(x>5, axis=1))
row

# %%
# Get filled the diagonals of NumPy array
x = np.array([[0,3,2],[5,0,7],[8,9,0]])
np.fill_diagonal(x, 1)
x

# %%
# Check elements present in the NumPy array
print(0 in x)

# %%
# Linear Algebra
# %%
# Find a matrix or vector norm using NumPy
vec = np.arange(10)
print(vec)
norm = np.linalg.norm(vec)
print(norm)

# %%
# Calculate the QR decomposition of a given matrix using NumPy
matrix1 = np.array([[1, 2, 3], [3, 4, 5]])
q, r = np.linalg.qr(matrix1)
print('\n',q,'\n',r)

# %%
# Compute the condition number of a given matrix using NumPy
result = np.linalg.cond(matrix1)
print(result)

# %%
# Compute the eigenvalues and right eigenvectors of a given square array using NumPy?
matrix1 = np.array([[1, 2, 3], [3, 4, 5],[6,7,8]])
w, v = np.linalg.eig(matrix1)
print(w,'\n',v)

# %%
# Calculate the Euclidean distance using NumPy
p1 = np.array((1,2,3))
p2 = np.array((1,1,1))
dist = np.linalg.norm(p1-p2)
print(dist)

# %%
# Numpy Random
# %%
# Create a Numpy array with random values
n = np.rint(np.random.rand(10)*10)
n

# %%
# How to choose elements from the list with different probability using NumPy?
numlist = np.random.choice(n, 2, p=[0.01,0.19,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,])
print(numlist)

# %%
# How to get weighted random choice in Python?
import random
sampleList = [100, 200, 300, 400, 500]
randomList = random.choices(sampleList, weights=(10, 20, 30, 40, 50), k=5)
print(randomList)

# %%
#  numpy.random.uniform(low = 0.0, high = 1.0, size = None) 
np.rint(np.random.uniform(low = 0.0, high = 1.0, size = 4)*10) 

# %%
# Get Random Elements form geometric distribution
import matplotlib.pyplot as plt
gfg = np.random.geometric(0.65, 1000)
count, bins, ignored = plt.hist(gfg, 40, density = True)
plt.show()

# %%
# Get Random elements from Laplace distribution
gfg = np.random.laplace(1.45, 15, 1000)
count, bins, ignored = plt.hist(gfg, 30, density = True)
plt.show()

# %%
# Return a Matrix of random values from a uniform distribution
np.rint(np.random.uniform(low = 0.0, high = 1.0, size = (4,3))*10)

# %%
# Return a Matrix of random values from a Gaussian distribution
np.rint(np.random.normal(size = (4,3))*10)

# %%
# Numpy sorting and searching
# %%
# How to get the indices of the sorted array using NumPy in Python?
arr = np.array([10, 52, 62, 16, 16, 54, 453])
arr = np.unique(arr)
indArr = np.argsort(arr, axis=-1, kind="quicksort", order=None)
print(indArr)
arr = arr[indArr]
print(arr)

# %%
# Finding the k smallest values of a NumPy array
k = 4
arr[:k]

# %%
# How to get the n-largest values of an array using NumPy?
n = 2
arr[-2:]

# %%
# Sort the values in a matrix
matrix1 = np.array([[3,4,2],[7,5,9]])
matrix1.sort()
print(matrix1)

# %%
# Filter out integers from float numpy array
floatAndInt = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
result = floatAndInt[floatAndInt != floatAndInt.astype(int)]
print(result)

# %%
# Numpy mathematics
# %%
# How to get element-wise true division of an array using Numpy?
result = np.true_divide(matrix1, 2)
print(result)

# %%
# How to calculate the element-wise absolute value of NumPy array?
arr = np.array([1,-2,3])
result = np.absolute(arr)
print(result)

# %%
# Compute the negative of the NumPy array
arr = np.negative(arr)
print(arr)

# %%
# Multiply 2d numpy array corresponding to 1d array
x = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
y = np.array([0, 2, 3])
result = x * y[:,np.newaxis]
print(result)

# %%
# Compute the nth percentile of the NumPy array
n = 25
np.percentile(arr,n)

# %%
# Calculate the n-th order discrete difference along the given axis
n = 1
np.diff(arr, n)

# %%
# Calculate the sum of all columns in a 2D NumPy array
np.sum(matrix1, axis = 0)

# %%
# Calculate average values of two given NumPy arrays
avg = ((arr + matrix1)/2)
print(avg)

# %%
# How to get the floor, ceiling and truncated values of the elements of a numpy array?
matrix1 = np.array([-1.8, -1.6, -0.5, 0.5,1.6, 1.8, 3.0])
print(np.floor(matrix1))
print(np.ceil(matrix1))
print(np.trunc(matrix1))

# %%
# How to round elements of the NumPy array to the nearest integer?
np.trunc(matrix1)

# %%
# Find the round off the values of the given matrix
matrix1.round()

# %%
# Determine the positive square-root of an array
matrix1 = np.array([[1, 4, 9, 16],[36, 100, 121, 400]])
sqrMatrix = np.sqrt(matrix1)
print(sqrMatrix)

# %%
# Evaluate Einstein’s summation convention of two multidimensional NumPy arrays
np.einsum("mk,nk",matrix1, sqrMatrix)

# %%
# Statistics
# %%
# Compute the median of the flattened NumPy array
flatten = np.array([[2,3,4],[3,6,8]]).flatten()
print(flatten)
print(np.median(flatten))

# %%
# Find Mean of a List of Numpy Array
inp = [np.array([1, 2, 3]),np.array([4, 5, 6]),np.array([7, 8, 9])]
out = []
for i in range(len(inp)):
    out.append(np.mean(inp[i]))
print(out)

# %%
# Calculate the mean of array ignoring the NaN value
arr = np.array([[20, 15, 37], [47, 13, np.nan]])
print(np.nanmean(arr))

# %%
# Get the mean value from given matrix
np.mean(inp)

# %%
# Compute the variance of the NumPy array
np.var(inp)

# %%
# Compute the standard deviation of the NumPy array
np.std(inp)

# %%
# Compute pearson product-moment correlation coefficients of two given NumPy arrays
np.corrcoef(inp)

# %%
# Calculate the average, variance and standard deviation in Python using NumPy
np.average(inp)
np.var(inp)
np.std(inp)

# %%
# Describe an array
arr = np.array([4, 5, 8, 5, 6, 4,9, 2, 4, 3, 6])
min = np.amin(arr)
max = np.amax(arr)
range = np.ptp(arr)
variance = np.var(arr)
sd = np.std(arr)
 
print("Array =", arr)
print("Measures of Dispersion")
print("Minimum =", min)
print("Maximum =", max)
print("Range =", range)
print("Variance =", variance)
print("Standard Deviation =", sd)

# %%
# Polynomial
# %%
# Defining a polynomial function
p1 = np.poly1d([5,-2,5])
print(p1)
print(p1(2))

# %%
# How to add one polynomial to another using NumPy in Python?
p2 = np.poly1d([2,-5,2])
print(p2)
result = np.polynomial.polynomial.polyadd(p1,p2)
print(result)

# %%
# How to subtract one polynomial to another using NumPy in Python?
result = np.polynomial.polynomial.polysub(p1,p2)
print(result)

# %%
# How to multiply a polynomial to another using NumPy in Python?
result = np.polynomial.polynomial.polysub(p1,p2)
print(result)

# %%
# How to divide a polynomial to another using NumPy in Python?
result = np.polynomial.polynomial.polydiv(p1,p2)
print(result)

# %%
# Find the roots of the polynomials using NumPy
roots = p1.r
print(roots)
coef=[1,2,1]
roots = np.roots(coef)
print(roots)
# %%
# Evaluate a 2-D polynomial series on the Cartesian product
c = np.array([[1, 3, 5], [2, 4, 6]]) 
ans = np.polynomial.polynomial.polygrid2d([7, 9], [8, 10], c)
print(ans)

# %%
# Evaluate a 3-D polynomial series on the Cartesian product
c = np.arange(24).reshape(2,2,3,2)
print(np.polynomial.polynomial.polyval3d([2,1],[1,2],[2,3], c))

# %%
# Numpy Strings
# %%
# Repeat all the elements of a NumPy array of strings
arr = np.array(['Akash', 'Rohit', 'Ayush','Dhruv', 'Radhika'], dtype = str)
newArr = np.char.multiply(arr,3)
print(newArr)

# %%
# How to split the element of a given NumPy array with spaces?
array = np.array(['Mai rapai, eh mermo'], dtype=str)
splited = np.char.split(array)
print(splited)

# %%
# How to insert a space between characters of all the elements of a given NumPy array?
x = np.array(["eh", "nada", "menino"], dtype=str)
result = np.char.join(" ", x)
print(result)

# %%
# Find the length of each string element in the Numpy array
arr = np.array(['New York', 'Lisbon', 'Beijing', 'Quebec'], dtype=str)
print(arr)
lenght = np.vectorize(len)
arr_len = lenght(arr)
print(arr_len)

# %%
# Swap the case of an array of string
array = np.char.swapcase(array)
print(array)
# %%
# Change the case to uppercase of elements of an array
upper = np.char.upper(array)
print(upper)

# %%
# Change the case to lowercase of elements of an array
lower = np.char.lower(array)
print(lower)

# %%
# Join String by a seperator
result = np.core.defchararray.join('-',arr)
print(result)

# %%
# Check if two same shaped string arrays one by one
result = np.char.equal(lower, upper)
print(result)

# %%
# Count the number of substrings in an array
result = np.char.count(lower, sub="ai")
print(result)

# %%
# Find the lowest index of the substring in an array
result = np.char.find(lower, sub="ai")
print(result)

# %%
# Get the boolean array when values end with a particular character
result = np.char.endswith(lower, 'mo')
print(result)

# %%
# More Questions
# %%
# Different ways to convert a Python dictionary to a NumPy array

# %%
# How to convert a list and tuple into NumPy arrays?

# %%
# Ways to convert array of strings to array of floats

# %%
# Convert a NumPy array into a csv file

# %%
# How to Convert an image to NumPy array and save it to CSV file using Python?

# %%
# How to save a NumPy array to a text file?

# %%
# Load data from a text file

# %%
# Plot line graph from NumPy array

# %%
# Create Histogram using NumPy
# %%
