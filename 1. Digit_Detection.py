import numpy as np
import idx2numpy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import matplotlib.pyplot as plt

# 1. LOADING AND MANIPULATION OF DATA
print("1. LOADING AND MANIPULATION OF DATA--------------------------")
""" 1. The images file consists of 60,000 training images , of handrawn digits where each image is represented in the form of 28*28 pixels 
    and each pixel brightness can be represented in the form of digit from [0,255], the entire set can be represented as a 3D array 
    --> (60000=Total images, 28=Height, 28=Width)
    2.The labels is a 1D array which shows which number that handrawn pictures represent, its value lies in the range [0,9], we are considering
    that only one digit is represented by an image
    -->(60000)=Total images
    
"""
# Paths to files
images_path = "C:/Users/vtyag/Downloads/archive/train-images.idx3-ubyte"
labels_path = "C:/Users/vtyag/Downloads/archive/train-labels.idx1-ubyte"

# Load data
X_train = idx2numpy.convert_from_file(images_path)
y_train = idx2numpy.convert_from_file(labels_path) 

print(f"Train images shape: {X_train.shape}")
print(f"Train labels shape: {y_train.shape}")

X_train = X_train.reshape(X_train.shape[0], -1)  # Convert (60000, 28, 28) → (60000, 784) flattened vectors so that they can be easily read

# Selecting only digits 0, 1, and 2 as instructed
provided_classes = [0, 1, 2]
bool_array = np.isin(y_train, provided_classes)# Provides a boolean array with yes at indexes where the label is 0,1,2
X_train_filtered = X_train[bool_array] #only add those images where label is given in provided classes
y_train_filtered = y_train[bool_array]

print(f"Filtered dataset shape: {X_train_filtered.shape}, Labels shape: {y_train_filtered.shape}")
X_train_filtered = X_train_filtered.astype(np.float32) / 255.0
 # Scale pixels to range [0, 1] by dividing each value by 255



# Splitting the Filtered Dataset into Train and Test
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(
    X_train_filtered, y_train_filtered,
    test_size=0.5,
    stratify=y_train_filtered,
    random_state=42
)

# Selecting Exactly 100 Samples per Class
train_indices = []#storing indexes
test_indices = []
for c in provided_classes:
    train_indices.extend(np.where(y_train_sampled == c)[0][:100])#finds all the indices in the training subset where the label is c
    test_indices.extend(np.where(y_test_sampled == c)[0][:100])

X_train_final = X_train_sampled[train_indices]
y_train_final = y_train_sampled[train_indices]
X_test_final = X_test_sampled[test_indices]
y_test_final = y_test_sampled[test_indices]

print(f"Final Train shape: {X_train_final.shape}, Final Test shape: {X_test_final.shape}")
print("\n")





#--------------------------------------------------------------------------------------------------------------------------------------
# 2. MAXIMUM LIKELIHOOD CALCULATION
print("2. MAXIMUM LIKELIHOOD CALCULATION----------------------------")


classes = np.unique(y_train_final)

means = {}        # Dictionary to store mean vectors, keyed by class
covariances = {}  # Dictionary to store covariance matrices, keyed by class

# 2A. Compute the Mean Vector for Each Class
for c in classes:
    # Selecting all samples for class c
    class_samples = X_train_final[y_train_final == c]  # shape: (N_c, 784)
    Nc = len(class_samples)                               
    mean_vector = np.zeros(784)
    for i in range(Nc):
        mean_vector += class_samples[i]
    mean_vector /= Nc

    means[c] = mean_vector
    print(f"Mean vector for class {c} computed with shape: {mean_vector.shape}")

# 2B. Computing the Covariance Matrix for Each Class
for c in classes:
    # Reuse the same subset of samples
    class_samples = X_train_final[y_train_final == c]  # shape: (N_c, 784)
    Nc = len(class_samples)

    # Reshape mean vector to (784, 1) for matrix operations
    mean_vector = means[c].reshape(784, 1)

    # Initialize a 784×784 matrix of zeros
    covariance_matrix = np.zeros((784, 784))
    for i in range(Nc):
        xi = class_samples[i].reshape(784, 1)  # (784,1)
        diff = xi - mean_vector                # (784,1)
        covariance_matrix += np.dot(diff, diff.T)  # (784,784)

    # For MLE, divide by Nc
    covariance_matrix /= Nc

    covariances[c] = covariance_matrix
    print(f"Covariance matrix for class {c} computed with shape: {covariance_matrix.shape}")

# # 2C. Display a Partial Representation
# for c in classes:
#     print(f"\n=== Class {c} ===")

#     # Show first 10 entries of the mean vector
#     print("Mean vector sample (first 200 values):")
#     print(means[c][:200])

#     # Show top-left 5×5 submatrix of the covariance matrix
#     print("\nCovariance matrix sample (top-left 5×5):")
#     print(covariances[c][:5, :5])
print("\n")



#------------------------------------------------------------------------------------------------------------------------------------------------
#3. PRINCIPAL COMPONENT ANALYSIS
print("3. PRINCIPAL COMPONENT ANALYSIS-------------------------------")

#3A. You need to obtain data matrix X ∈ R784×300. Then obtain its mean μ.
X = X_train_final.T  #(784, 300)
print(f"Shape of data matrix X: {X.shape}")
# Compute mean vector manually
μ = np.zeros((784, 1))# Column vector of size 784x1
N = X.shape[1]# Number of samples 
for i in range(N):
    μ += X[:, i].reshape(784, 1)  
μ /= N  # Divide by the number of samples
print(f"Mean vector shape: {μ.shape}")  #(784, 1)

#3B. Remove the mean form X to obtain Xc.
# Subtracting mean from X
Xc = X - μ
print(f"Mean-centered data shape: {Xc.shape}")  #(784, 300)

#3C.  Obtain covariance S=XcX>c /(300 − 1).
S = (Xc @ Xc.T) / (N - 1)
print(f"Covariance matrix shape: {S.shape}")  #(784, 784)

#3D. Obtain eigenvectors and eigenvalues. Sort the eigenvectors in descending order of eigenvalues.
# Computing eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(S)

# Sorting eigenvectors in descending order of eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]  
eigenvalues = eigenvalues[sorted_indices]  # Sort eigenvalues
eigenvectors = eigenvectors[:, sorted_indices]  # Sort eigenvectors accordingly

print(f"Eigenvalues shape: {eigenvalues.shape}")  # (784,)
print(f"Eigenvectors shape: {eigenvectors.shape}")  # (784, 784)

#3E. Obtain Up and perform Y = Upt Xc.
# Compute cumulative variance ratio
total_variance = np.sum(eigenvalues)
explained_variance_ratio = np.cumsum(eigenvalues) / total_variance

# Find minimum number of components to retain 95% variance
p = np.argmax(explained_variance_ratio >= 0.95) + 1
print(f"Number of principal components selected: {p}")

# Selecting top p eigenvectors
Up = eigenvectors[:, :p]  # (784, p)
# Projecting data to lower dimensions
Y = Up.T @ Xc  # Now (p, 300)
print(f"Transformed data shape: {Y.shape}")  #(p, 300)
def transform_new_sample(x_test):
    x_test = x_test.reshape(784, 1)  # Convert to column vector
    x_test_centered = x_test - μ  # Subtract mean
    y_test = Up.T @ x_test_centered  # Project to lower dimensions
    return y_test
# Example transformation
sample_test = X_test_final[0] 
y_test_sample = transform_new_sample(sample_test)
print(f"Transformed test sample shape: {y_test_sample.shape}")  #(p, 1)
print("\n")




#------------------------------------------------------------------------------------------------------------------------------------------------
#4. FISHERS DISCRIMINANT ANALYSIS
print("4. FISHERS DISCRIMINANT ANALYSIS-------------------------------")

# Computing class means
means_fda = {}  # Dictionary to store mean vectors
classes = np.unique(y_train_final)  # Unique classes [0, 1, 2]

for c in classes:
    class_samples = X_train_final[y_train_final == c]  # Select samples for class c
    means_fda[c] = np.mean(class_samples, axis=0).reshape(784, 1)  # Compute mean

    print(f"Mean vector for class {c} shape: {means_fda[c].shape}")  #(784, 1)
# Computing overall mean
μ_fda = np.mean(X_train_final, axis=0).reshape(784, 1)  # 
print(f"Overall mean vector shape: {μ_fda.shape}")  # (784, 1)

# Computing Scatter Matrix
SB = np.zeros((784, 784))  # Initialize matrix
for c in classes:
    Nc = np.sum(y_train_final == c)  # Number of samples in class c
    mean_diff = means_fda[c] - μ_fda  # (μc - μ)
    SB += Nc * (mean_diff @ mean_diff.T)  # Nc(μc - μ)(μc - μ)^T

print(f"Between-Class Scatter Matrix shape: {SB.shape}")  # Should be (784, 784)

# Compute Within-Class Scatter Matrix (SW)
SW = np.zeros((784, 784))  # Initialize matrix

for c in classes:
    class_samples = X_train_final[y_train_final == c]  # Select samples for class c
    mean_c = means_fda[c]  # Class mean
    
    for xi in class_samples:
        xi = xi.reshape(784, 1)  # Convert to column vector
        diff = xi - mean_c  # (xi - μc)
        SW += diff @ diff.T  # (xi - μc)(xi - μc)^T



print(f"Within-Class Scatter Matrix shape: {SW.shape}")  # Should be (784, 784)

alpha = 1e-5
SW_reg = SW + alpha * np.eye(784)
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(SW_reg) @ SB)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigvals)[::-1]
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]

print(f"Eigenvectors shape: {eigvecs.shape}")
# Select top 2 eigenvectors for 3-class problem
W_fda = eigvecs[:, :2]  # (784, 2)
print(f"Optimal projection matrix W shape: {W_fda.shape}")  #  (784, 2)

# Project training and test data onto the new FDA space
X_train_fda = W_fda.T @ X_train_final.T  # (2, 300)
X_test_fda = W_fda.T @ X_test_final.T  # (2, 300)

print(f"Projected Train Data shape: {X_train_fda.shape}")  # (2, 300)
print(f"Projected Test Data shape: {X_test_fda.shape}")  # (2, 300)

#------------------------------------------------------------------------------------------------------------------------------------------------
#5. EVALUATION AND COMPARISON
print("5. EVALUATION AND COMPARISON--------------------------------")
def accuracy_score(y_true, y_pred):
    """
    Returns the fraction of correct predictions.
    """
    return np.mean(y_true == y_pred)

def compute_class_means_and_covs(X, y):
    """
    Compute the MLE mean vector and covariance matrix for each class
    (for QDA).
    Also compute the prior p(class=c).
    
    We ensure covariances are forced to real floats to avoid
    complex-valued arrays.
    """
    classes_local = np.unique(y)
    means = {}
    covs = {}
    priors = {}
    N = len(y)
    
    for c in classes_local:
        X_c = X[y == c]
        Nc = len(X_c)
        priors[c] = Nc / N
        
        # Mean
        mu_c = np.mean(X_c, axis=0)  # shape (d,)
        
        # Covariance (MLE)
        diff = X_c - mu_c
        cov_c = (diff.T @ diff) / Nc
        
        # Force real float in case of tiny imaginary parts
        cov_c = np.real_if_close(cov_c, tol=1000)
        cov_c = cov_c.astype(float)
        
        means[c] = mu_c.astype(float)
        covs[c] = cov_c
    
    return means, covs, priors

def shared_covariance(covs, priors, classes_local):
    """
    For LDA, we assume a single shared covariance:
       Sigma = sum( priors[c] * covs[c] )
    """
    d = list(covs.values())[0].shape[0]
    Sigma = np.zeros((d, d), dtype=float)
    for c in classes_local:
        # Make sure each covs[c] is float
        cov_float = np.real_if_close(covs[c], tol=1000).astype(float)
        Sigma += priors[c] * cov_float
    return Sigma

def predict_LDA(X, means, Sigma, priors, classes_local):
    """
    Manual LDA classification:
      discriminant ~ -1/2 (x - mu)^T Sigma^-1 (x - mu) + ln(prior_c)
    X shape: (N, d)
    """
    inv_Sigma = np.linalg.inv(Sigma)
    y_pred = []
    for x in X:
        best_class = None
        best_score = -np.inf
        for c in classes_local:
            diff = (x - means[c]).reshape(-1,1)
            score = -0.5 * (diff.T @ inv_Sigma @ diff) + np.log(priors[c])
            if score > best_score:
                best_score = score
                best_class = c
        y_pred.append(best_class)
    return np.array(y_pred)

def predict_QDA(X, means, covs, priors, classes_local):
    """
    Manual QDA classification:
      discriminant ~ -1/2 ln|Sigma_c| -1/2 (x - mu_c)^T Sigma_c^-1 (x - mu_c) + ln(prior_c)
    """
    inv_covs = {}
    log_dets = {}
    for c in classes_local:
        cov_float = np.real_if_close(covs[c], tol=1000).astype(float)
        inv_covs[c] = np.linalg.inv(cov_float)
        sign, logdet_val = np.linalg.slogdet(cov_float)
        log_dets[c] = logdet_val  # sign should be +1 if cov is pos. def.
    
    y_pred = []
    for x in X:
        best_class = None
        best_score = -np.inf
        for c in classes_local:
            diff = (x - means[c]).reshape(-1,1)
            score = -0.5 * log_dets[c] - 0.5*(diff.T @ inv_covs[c] @ diff) + np.log(priors[c])
            if score > best_score:
                best_score = score
                best_class = c
        y_pred.append(best_class)
    return np.array(y_pred)


#   5A. FDA + LDA/QDA

X_train_fda_T = X_train_fda.T  # shape (N_train, 2)
X_test_fda_T = X_test_fda.T    # shape (N_test, 2)

# Compute means/covs in the 2D FDA space
means_fda_2d, covs_fda_2d, priors_fda_2d = compute_class_means_and_covs(X_train_fda_T, y_train_final)
Sigma_fda_2d = shared_covariance(covs_fda_2d, priors_fda_2d, classes)

# LDA on FDA
y_train_lda_fda = predict_LDA(X_train_fda_T, means_fda_2d, Sigma_fda_2d, priors_fda_2d, classes)
y_test_lda_fda = predict_LDA(X_test_fda_T, means_fda_2d, Sigma_fda_2d, priors_fda_2d, classes)
train_acc_lda_fda = accuracy_score(y_train_final, y_train_lda_fda) * 100
test_acc_lda_fda = accuracy_score(y_test_final, y_test_lda_fda) * 100

# QDA on FDA
y_train_qda_fda = predict_QDA(X_train_fda_T, means_fda_2d, covs_fda_2d, priors_fda_2d, classes)
y_test_qda_fda = predict_QDA(X_test_fda_T, means_fda_2d, covs_fda_2d, priors_fda_2d, classes)
train_acc_qda_fda = accuracy_score(y_train_final, y_train_qda_fda) * 100
test_acc_qda_fda = accuracy_score(y_test_final, y_test_qda_fda) * 100

print("5A.  FDA + LDA/QDA ")
print(f"LDA on FDA -> Train: {train_acc_lda_fda:.2f}%, Test: {test_acc_lda_fda:.2f}%")
print(f"QDA on FDA -> Train: {train_acc_qda_fda:.2f}%, Test: {test_acc_qda_fda:.2f}%\n")

#   5B. PCA -> LDA (95%, 90%, 2 Components)

def manual_pca(X, var_threshold=None, n_components=None):
    N, d = X.shape
    meanX = np.mean(X, axis=0)
    Xc = X - meanX  # center data
    
    # Covariance (N-1 for unbiased)
    S = (Xc.T @ Xc) / (N - 1)
    eigvals, eigvecs = np.linalg.eig(S)
    
    # Sort descending
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]
    
    # Force real
    eigvals = np.real_if_close(eigvals, tol=1000).astype(float)
    eigvecs = np.real_if_close(eigvecs, tol=1000).astype(float)
    
    if n_components is not None:
        p = n_components
    else:
        # find p for var_threshold
        cum_var = np.cumsum(eigvals) / np.sum(eigvals)
        p = np.searchsorted(cum_var, var_threshold) + 1
    
    U_p = eigvecs[:, :p]  # top p eigenvectors
    X_pca = Xc @ U_p      # project data
    return X_pca, U_p, meanX

def manual_pca_transform(X, U_p, meanX):
    return (X - meanX) @ U_p

def evaluate_pca_lda(X_train, y_train, X_test, y_test, var_threshold=None, n_components=None):
    
    # PCA
    X_train_pca, U_p, meanX = manual_pca(X_train, var_threshold, n_components)
    X_test_pca = manual_pca_transform(X_test, U_p, meanX)
    
    # Means/covs in the p-dim PCA space
    means_pca, covs_pca, priors_pca = compute_class_means_and_covs(X_train_pca, y_train)
    Sigma_pca = shared_covariance(covs_pca, priors_pca, np.unique(y_train))
    
    # LDA in PCA space
    y_train_lda = predict_LDA(X_train_pca, means_pca, Sigma_pca, priors_pca, np.unique(y_train))
    y_test_lda = predict_LDA(X_test_pca, means_pca, Sigma_pca, priors_pca, np.unique(y_train))
    
    train_acc = accuracy_score(y_train, y_train_lda) * 100
    test_acc = accuracy_score(y_test, y_test_lda) * 100
    return train_acc, test_acc


train_acc_95, test_acc_95 = evaluate_pca_lda(X_train_final, y_train_final,X_test_final, y_test_final,var_threshold=0.95, n_components=None)
print("5B.  PCA (95% variance) + LDA ")
print(f"Train Acc: {train_acc_95:.2f}%, Test Acc: {test_acc_95:.2f}%\n")

train_acc_90, test_acc_90 = evaluate_pca_lda(X_train_final, y_train_final,
                                             X_test_final, y_test_final,
                                             var_threshold=0.90, n_components=None)
print("5C.  PCA (90% variance) + LDA ")
print(f"Train Acc: {train_acc_90:.2f}%, Test Acc: {test_acc_90:.2f}%\n")


train_acc_2, test_acc_2 = evaluate_pca_lda(X_train_final, y_train_final,
                                           X_test_final, y_test_final,
                                           var_threshold=None, n_components=2)
print("5D. PCA (2 Components) + LDA ")
print(f"Train Acc: {train_acc_2:.2f}%, Test Acc: {test_acc_2:.2f}%\n")




