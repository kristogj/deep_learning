import numpy as np
import matplotlib.pyplot as plt

a = [3.2, 2.8, 2.2]

x = np.arange(1, len(a) + 1, 1)


plt.plot(x, a)
plt.xticks(x)
plt.show()