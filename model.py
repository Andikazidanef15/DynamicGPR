# Import Package
import numpy as np
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from kernels import rbf_kernel, ard_kernel

class DynamicGPR():
    def __init__(self, l_init, sigma_init, weight_init, 
               n_clusters, learning_rate, kernel, batch_size = 1):
        self.l = l_init
        self.sigma = sigma_init 
        self.weight = weight_init
        self.n_clusters = n_clusters 
        self.learning_rate = learning_rate
        self.kernel_type = kernel
    
        if self.kernel_type == 'RBF':
            self.kernel = rbf_kernel(l = self.l, sigma = self.sigma)
        elif self.kernel_type == 'ARD':
            self.kernel = ard_kernel(weight = self.weight, sigma = self.sigma)
        
        self.batch_size = batch_size
    
    def fit(self, X_train, y_train, eval_data = None):
        # Check wheter X_train n_dim is 1 
        if X_train.ndim == 1:
            X_train_mat = X_train.reshape(-1, 1)
        else:
            X_train_mat = X_train
    
        # Buat n buah cluster dari datanya lalu ambil titik tengah tiap cluster
        kmeans = KMeans(n_clusters = self.n_clusters, random_state = 111).fit(X_train_mat)
        self.center_clusters = kmeans.cluster_centers_

        # Inisiasi kernel, S, dan m
        self.S = self.kernel.fit(self.center_clusters, self.center_clusters)
        self.m = np.zeros((self.n_clusters, 1))

        # Set trange
        t = trange(0, X_train.shape[0], self.batch_size, desc = 'ML')

        for n in t:
            # Update m dan S 
            # ----------------------------------------
            # Inisiasi Kernel(X,X), Kernel(X,C), Kernel(C,X), q^T, q, K
            obs = X_train_mat[n : n + self.batch_size, :]
            kernel_x_x = self.kernel.fit(obs, obs)
            kernel_x_c = self.kernel.fit(obs, self.center_clusters)
            kernel_c_x = self.kernel.fit(self.center_clusters, obs)
            K = self.kernel.fit(self.center_clusters, self.center_clusters)
            K = K + 0.01*np.identity(K.shape[0])

            # Cek apakah matriks invers K singular, jika iya, tambahkan dengan matriks identitas
            K_inv = np.linalg.inv(K)

            # Hitung cov_x_x dan cov_c_x
            sum_1_x_x = kernel_x_c @ K_inv @ kernel_c_x
            sum_1_x_c = kernel_x_c @ K_inv @ K
            sum_1_c_x = K @ K_inv @ kernel_c_x
            sum_2_x_x = kernel_x_c @ K_inv @ self.S @ K_inv @ kernel_c_x 
            sum_2_x_c = kernel_x_c @ K_inv @ self.S @ K_inv @ K 
            sum_2_c_x = K @ K_inv @ self.S @ K_inv @ kernel_c_x 

            cov_x_x = kernel_x_x - sum_1_x_x + sum_2_x_x 
            cov_x_x = cov_x_x + 0.01*np.identity(cov_x_x.shape[0])
            cov_x_c = kernel_x_c - sum_1_x_c + sum_2_x_c
            cov_c_x = kernel_c_x - sum_1_c_x + sum_2_c_x

            # Hitung mu
            mu = kernel_x_c @ K_inv @ self.m

            # Update m dan S
            self.m = self.m + (cov_c_x @ np.linalg.inv(cov_x_x) @ (y_train[n : n + self.batch_size].reshape(-1, 1) - mu))
            self.S = self.S - (cov_c_x @ np.linalg.inv(cov_x_x) @ cov_x_c)

            # GRADIENT DESCENT
            # -------------------------------
            # Tentukan alpha dan partial differential l serta sigma
            alpha = K_inv @ self.m
            if self.kernel_type == 'RBF':
                partial_diff_sigma, partial_diff_l = self.kernel.partial_diff(self.center_clusters, self.center_clusters)

                # Hitung diff sigma dan diff l
                mat_mul_sigma = (alpha @ np.transpose(alpha) - K_inv) @ partial_diff_sigma
                mat_mul_l = (alpha @ np.transpose(alpha) - K_inv) @ partial_diff_l
                diff_sigma = (1/2)*np.trace(mat_mul_sigma)
                diff_l = (1/2)*np.trace(mat_mul_l)

                # Update theta
                self.l = self.l - self.learning_rate*diff_l
                self.sigma = self.sigma - self.learning_rate*diff_sigma

                # Update kernel
                self.kernel = rbf_kernel(l = self.l, sigma = self.sigma)

            elif self.kernel_type == 'ARD':
                partial_diff_sigma, partial_diff_weight = self.kernel.partial_diff(self.center_clusters, self.center_clusters)

                # Update sigma
                mat_mul_sigma = (alpha @ np.transpose(alpha) - K_inv) @ partial_diff_sigma
                diff_sigma = (1/2)*np.trace(mat_mul_sigma)
                self.sigma = self.sigma - self.learning_rate*diff_sigma

                # Update weight
                for n, weight_mat in enumerate(partial_diff_weight):
                    mat_mul_weight = (alpha @ np.transpose(alpha) - K_inv) @ weight_mat
                    diff_weight = (1/2)*np.trace(mat_mul_weight)
                    self.weight[n] = self.weight[n] - self.learning_rate*diff_weight

                # Update kernel
                self.kernel = ard_kernel(weight = self.weight, sigma = self.sigma)
                
    def predict(self, X_test, center_cluster = False, eval_kernel = False, return_var = False):
        # Check dimention of X_test
        if X_test.ndim == 1:
            X_test_mat = X_test.reshape(-1, 1)
        else:
            X_test_mat = X_test
  
        # Inisiasi Kernel(X,C), q^T, q
        kernel_x_c = self.kernel.fit(X_test_mat, self.center_clusters)
        kernel_c_x = self.kernel.fit(self.center_clusters, X_test_mat)
        q_transpose = self.kernel.fit(X_test_mat, self.center_clusters)
        q = np.transpose(q_transpose)
        K = self.kernel.fit(self.center_clusters, self.center_clusters)
        K = K + 0.01*np.identity(K.shape[0])
        
        if return_var == True:
            # Inisiasi Kernel(X,X)
            kernel_x_x = self.kernel.fit(X_test_mat, X_test_mat)
            # Hitung perkalian matriks pada rumus var_pred
            #sum_1 = kernel_x_c @ K_inv @ kernel_c_x
            #sum_2 = kernel_x_c @ K_inv @ self.S @ K_inv @ kernel_c_x
            sum_1 = kernel_x_c @ K_inv @ kernel_c_x
            sum_2 = kernel_x_c @ K_inv @ self.S @ K_inv @ kernel_c_x
            var_pred = kernel_x_x - sum_1 + sum_2
            
        # Cek apakah matriks invers K singular, jika iya, tambahkan dengan matriks identitas
        K_inv = np.linalg.inv(K)
    
        # Predict
        y_pred = q_transpose @ K_inv @ self.m
        y_pred = y_pred.flatten()
    
        if center_cluster == False:
            if eval_kernel == True:
                if self.kernel_type == 'RBF':
                    print('RBF Optimized')
                    print('Optimized length scale:', self.l)
                    print('Optimized sigma signal:', self.sigma)
                elif self.kernel_type == 'ARD':
                    print('ARD Optimized')
                    print('Optimized weight:', self.weight)
                    print('Optimized sigma signal:', self.sigma)
                    
            if return_var == True:    
                return y_pred, var_pred
            else:
                return y_pred
        else:
            if eval_kernel == True:
                if self.kernel_type == 'RBF':
                    print('RBF Optimized')
                    print('Optimized length scale:', self.l)
                    print('Optimized sigma signal:', self.sigma)
                elif self.kernel_type == 'ARD':
                    print('ARD Optimized')
                    print('Optimized weight:', self.weight)
                    print('Optimized sigma signal:', self.sigma)
            if return_var == True:
                return y_pred, var_pred, self.center_clusters.flatten()
            else:
                return y_pred, self.center_clusters.flatten()
