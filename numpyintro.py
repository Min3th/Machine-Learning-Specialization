import numpy as np
import time

a = np.zeros(4)
print(a)
print(a.shape)
print(a.dtype)
b = np.random.random_sample(4)
print(b)

c = np.arange(4)
print(c)

d = np.array([3,2,5,4])
e = np.array([3.2,2.5,1,5])

print(d)
print(e)
print(e.dtype)

newa= np.arange(10)
print(newa)
print(newa[2].shape)

try:
  newc=newa[10]
except Exception as e:
  print("Error is: ")
  print(e)

f = np.array([1,2,3,4])

g = np.array([3,4,2,1])
print(np.dot(g,f))

def myDot(a,b):
  x=0
  for i in range(a.shape[0]):
    x=x+a[i]*b[i]
  return x

np.random.seed(1) #to ensure same random number is generated every time the code is run

# a=np.random.rand(10000000)
# b=np.random.rand(10000000)

# startTime = time.time()
# k = np.dot(a,b)
# endTime = time.time()

# print(1000*(endTime-startTime))

# startTime = time.time()
# k = myDot(a,b)
# endTime = time.time()

# print(1000*(endTime-startTime))

a = np.zeros((2,5))
print(a)

a=np.array([[5],[4],[2]])
print(a)

a = np.arange(12).reshape(3,-1)
print(a)

print(a[2,0:3:1])