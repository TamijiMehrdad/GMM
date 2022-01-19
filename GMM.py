
"""
Implementation of GMM
__author__ = "Mehrdad Tamiji"
__email__ = "mehrdad.tamiji@gmail.com"
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def dist_pdf(dist, mean, variance):
    """
    Calculate normal probability density function
    :param dist: Distributions
    :param mean: Mean of distribution
    :param variance: Variance of distribution
    :return: a vector of PDFs
    """
    return (1 / (np.sqrt( 2 * np.pi * variance))) * np.exp(- (np.power(dist - mean, 2) / (2 * variance)))

def show_plot(distributions, norm_dists_param, est_norm_dists_param, save_plot= False):
    """
    Show plot
    :param distributions: Combination of distributions
    :param norm_dists_param:
    :est_norm_dists_param
    :return:
    """
    bins = np.linspace(np.min(distributions), np.max(distributions), 1000, endpoint=True)
    plt.figure(figsize=(15, 8))
    plt.plot(bins, dist_pdf(bins, norm_dists_param[0, 0], norm_dists_param[0, 1]), color="black")
    plt.plot(bins, dist_pdf(bins, norm_dists_param[1, 0], norm_dists_param[1, 1]), color="black")
    plt.plot(bins, dist_pdf(bins, norm_dists_param[2, 0], norm_dists_param[2, 1]), color="black",label="Ground Truth")
    plt.plot(bins, dist_pdf(bins, est_norm_dists_param[0, 0], est_norm_dists_param[0, 1]), linestyle="--", color='red', label="Estimated Distribution1")
    plt.plot(bins, dist_pdf(bins, est_norm_dists_param[1, 0], est_norm_dists_param[1, 1]), linestyle="--", color='green', label="Estimated Distribution2")
    plt.plot(bins, dist_pdf(bins, est_norm_dists_param[2, 0], est_norm_dists_param[2, 1]), linestyle="--", color='blue', label="Estimated Distribution3")
    plt.legend(loc='upper left')
    if save_plot:
        plt.savefig(f"Mehrdad_Tamiji_Final_Result")
    plt.show()

def init():
    """
    Initialization of data and variables
    """
    distributions = np.empty(0)
    norm_dists_param = np.zeros((3,2))
    norm_dists_param[0] = np.array([-5, 4])
    norm_dists_param[1] = np.array([0, 4])
    norm_dists_param[2] = np.array([4, .04])
    dist1 = np.random.normal(norm_dists_param[0,0], np.sqrt(norm_dists_param[0,1]), 500)
    dist2 = np.random.normal(norm_dists_param[1,0], np.sqrt(norm_dists_param[1,1]), 1000)
    dist3 = np.random.normal(norm_dists_param[2,0], np.sqrt(norm_dists_param[2,1]), 500)
    distributions = np.hstack((distributions, dist1))
    distributions = np.hstack((distributions, dist2))
    distributions = np.hstack((distributions, dist3))
    np.random.shuffle(distributions)
    num_cluster = 3
    num_epoch = 2000
    weights = np.ones((num_cluster)) / num_cluster
    est_norm_dists_param = np.zeros((num_cluster, 2))
    kmeans_start = False  # to prevent converging two clusters to one distribution, but it is False, and we chose first means among data
    if kmeans_start == True:
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(distributions.reshape(-1, 1))
        est_norm_dists_param[:, 0] =  kmeans.cluster_centers_ #use kmeans for initialization of means
    else:
        est_norm_dists_param[:, 0] = np.random.choice(distributions, num_cluster)
    est_norm_dists_param[:, 1] = np.random.random_sample(size=num_cluster)
    return norm_dists_param, dist1, dist2, dist3, distributions, num_cluster, num_epoch, weights, est_norm_dists_param

def checking(distributions, norm_dists_param, est_norm_dists_param):
    """
    Difference between ground truth and estimation mean and variance and then show and save final results.
    """
    diff = norm_dists_param[norm_dists_param[:, 0].argsort()] - est_norm_dists_param[
        est_norm_dists_param[:, 0].argsort()]
    print(f"Difference between ground truth and estimation mean:\n {diff[:, 0]}")
    print(f"Difference between ground truth and estimation variance:\n {diff[:, 1]}")
    show_plot(distributions, norm_dists_param, est_norm_dists_param, True)

if __name__ == '__main__':
    norm_dists_param, dist1, dist2, dist3, distributions, num_cluster, num_epoch, weights, est_norm_dists_param = init()
    show_plot(distributions, norm_dists_param, est_norm_dists_param)
    for epoch in range(num_epoch):
        likelihood = np.zeros((3, distributions.shape[0]))

        for j in range(num_cluster):
            likelihood[j] = dist_pdf(distributions, est_norm_dists_param[j, 0], est_norm_dists_param[j, 1])
        coeff_weight = (likelihood * weights.reshape(-1, 1)) / (np.sum(likelihood * weights.reshape(-1, 1), axis=0) )
        est_norm_dists_param[:, 0] = np.sum(coeff_weight * distributions, axis=1) / (np.sum(coeff_weight, axis=1))
        est_norm_dists_param[:, 1] = np.sum(coeff_weight * np.square(distributions - est_norm_dists_param[:, 0].reshape(-1, 1)), axis=1) / (np.sum(coeff_weight, axis=1))
        weights = np.mean(coeff_weight, axis=1)
        if epoch % 999 == 0:
            show_plot(distributions, norm_dists_param, est_norm_dists_param)
    checking(distributions, norm_dists_param, est_norm_dists_param)