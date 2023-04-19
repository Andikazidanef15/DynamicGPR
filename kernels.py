# Import Packages
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class rbf_kernel():
    def __init__(self, l: float, sigma: float):
        self.l = l
        self.sigma = sigma
        self.kernel = (self.sigma**2) * RBF(self.l, length_scale_bounds = 'fixed')
    
    def fit(self, X, Y):
        return self.kernel.__call__(X, Y)
    
    def partial_diff(self, X, Y):
        # Hitung norm antar titik observasi
        norm_matrix = np.zeros([len(X), len(Y)])
        for i in range(len(X)):
            for j in range(len(Y)):
                norm_matrix[i, j] = np.linalg.norm(X[i] - Y[j])**2

        # Hitung partial differentialnya
        partial_sigma = RBF(self.l, length_scale_bounds = 'fixed').__call__(X, Y) 
        partial_l = (self.sigma/self.l**3)*norm_matrix @ RBF(self.l, 
                                                          length_scale_bounds = 'fixed').__call__(X, Y) 
        return partial_sigma, partial_l

class ard_kernel():
    def __init__(self, weight: np.array, sigma):
        self.weight = weight
        self.sigma = sigma
    
    def calculate_sum(self, X, Y):
        sum_weight = 0
        for j in range(len(X)):
            sum_weight += self.weight[j] * (X[j] - Y[j])**2
        return sum_weight
    
    def fit(self, X, Y):
        cov_matrix = np.zeros([len(X), len(Y)])
        for i in range(len(X)):
            for j in range(len(Y)):
                sum_of_feats = self.calculate_sum(X[i,:], Y[j,:])
                cov_matrix[i, j] = (self.sigma**2)*np.exp(-0.5*sum_of_feats)
        return cov_matrix
    
    def partial_diff(self, X, Y):
        partial_diff_sigma = np.zeros([len(X), len(Y)])
        partial_diff_weight = [np.zeros([len(X), len(Y)]) for i in range(X.shape[1])]
        
        for i in range(len(X)):
            for j in range(len(Y)):
                # Hitung partial diff sigma
                sum_of_feats = self.calculate_sum(X[i,:], Y[j,:])
                partial_diff_sigma[i, j] = np.exp(-0.5*sum_of_feats)
                
                # Hitung partial diff weight untuk setiap weight
                for n, weight_mat in enumerate(partial_diff_weight):
                    weight_mat[i, j] = -0.5*((X[i, n] - Y[j, n])**2)*np.exp(-0.5 * sum_of_feats)
                    partial_diff_weight[n] = weight_mat
                    
        return partial_diff_sigma, partial_diff_weight