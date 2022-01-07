import mtLeastR
import numpy as np

a = np.random.rand(10,10)
b = np.random.rand(10,1)
opts = {"maxIter":10,'ind':np.array([[10,1],[10,1]])}
x = mtLeastR.mtLeastR(a,b,0.01,opts)
print(x)