from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

a=np.arange(1,2001,1)

b=(1-1/(1+np.exp(-a/250)))*2*0.9+0.1



plt.figure()
plt.plot(a,b)
# plt.plot(ep)
plt.show()