import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


rv1 = norm(0, 0.5)
rv2 = norm(2, 0.5)

X = np.linspace(-2, 4, 500)
y = rv1.pdf(X) * 0.5 + rv2.pdf(X) * 0.5

plt.plot(X, y)
plt.show()