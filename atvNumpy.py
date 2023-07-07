# %%
import numpy as np
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
arr1 = np.array([1,2,3,4])
arr2 = np.array([[5,6,7,8], [9,10,11,12]])
arr3 = np.vstack((arr1,arr2))
print(arr3)

# %%
# creating an array of all combinations(2,2) of another two arrays
result = np.meshgrid(arr1[:3],arr2[0,:3])
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
print(arr2)
moved = np.moveaxis(arr2, 0, 1)
print(moved)

# %%
# interchanging axes
print(arr2)
changed = np.swapaxes(arr2, 0, 1)
print(changed)

# %%
# fibonacci series
