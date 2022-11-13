"""
# @Author: JuQi
# @Time  : 2022/11/12 16:34
# @E-mail: 18672750887@163.com
"""

import numpy as np

a = np.arange(24, dtype=np.float)
a = a.reshape((2, 2, 2, 3))

print(a)
b=np.mean(a,3)
print(b)

b=b.repeat(3)
print(b)
b=b.reshape((2,2,2,3))
print(a-b)
# b = np.array([0.1, 0.2])
# a[:, 0, :, :] = a[:, 0, :, :] * b[0]
# print(a)
