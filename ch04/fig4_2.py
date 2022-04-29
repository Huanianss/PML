import matplotlib.pyplot as plt
import numpy as np

z = np.linspace(-2, 2, 100)
loss_01 = z < 0
loss_hinge = [max(0, 1 - x) for x in z.tolist()]
loss_log = np.log2(1 + np.exp(-z))
loss_exp = np.exp(-z)


plt.plot(z, loss_01, 'k')
plt.plot(z, loss_hinge, 'b-')
plt.plot(z, loss_log, 'r--')
plt.plot(z, loss_exp, 'g')
plt.legend(['0-1 loss', 'hinge loss', 'log loss', 'exp loss'])
plt.axis([-2.1, 2.1, -0.1, 3.1])
plt.show()
