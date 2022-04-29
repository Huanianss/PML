import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# np.random.seed(1)
theta_star = 1
sigma_2 = 1
n = 5

data = norm(theta_star, sigma_2).rvs(n)
kappa0_ = np.linspace(0, 3, 4)
print(kappa0_)

x = np.linspace(-1, 2.5, 100)
style = ['b', 'r:', 'k-.', 'g--']
for (post, kappa0) in enumerate(kappa0_):
    prior_var = sigma_2 / (kappa0 + 1e-12)
    w = n / (n + kappa0)
    post_mean = w  # * np.mean(data)
    ###########################################
    '''
    samplingDist = []
    for i in range(10000):
        data_ = norm(post_mean, sigma_2).rvs(n)
        post_mean = np.mean(data_)
        samplingDist.append(post_mean)
    val, bins = np.histogram(samplingDist, bins=10, density=True)
    bins = (bins[0:-1] - bins[1:]) / 2 + bins[1:]
    plt.plot(bins,
             val,
             style[post],
             lw=2,
             label='postMean' + str(post))
    '''
    ###########################################
    post_var = sigma_2*w**2/n
    # Bayes
    Bayes_post_var =1/( 1/prior_var  +n/sigma_2)
    print(post_var,Bayes_post_var)
    pdf = norm(post_mean, post_var**0.5 ).pdf(x)
    plt.plot(x,
             pdf,
             style[post],
             lw=2,
             label='postMean' + str(post))

plt.title('sampling distribution, truth = ' + str(theta_star) + ', prior = 0, n = ' + str(n))
plt.legend()
plt.show()

ratio_ = []
data_ = norm(theta_star, sigma_2).rvs(n)
for n in range(1, 45):
    data = data_[0:n]
    # for kappa0 in kappa0_:
    # prior_var = sigma_2 / (kappa0 + 1e-12)

    post_mean = n / (n + kappa0_) * 1  # np.mean(data)
    w = n / (n + kappa0_)
    MSE_MAP = (w - 1) ** 2 + w ** 2 / n
    # MSE_MLE = (np.mean(data) - theta_star) ** 2
    MSE_MLE = 1 / n
    # print(n,MSE_MLE)
    ratio = MSE_MAP / MSE_MLE
    ratio_.append(ratio)
ratio = np.stack(ratio_)

print(ratio.shape)
for i in range(4):
    plt.plot(range(1, 45), ratio[:, i],
             style[i],
             lw=2,
             label='postMean' + str(i))
plt.title('MSE of postmean/MSE of MLE')
plt.xlabel('sample size')
plt.ylabel('relative MSE')
plt.legend()
plt.show()
