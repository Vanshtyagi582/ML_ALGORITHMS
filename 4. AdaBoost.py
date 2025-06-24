import os
import struct
import numpy as np
import matplotlib.pyplot as plt

TRAIN_IMAGES_PATH = r"C:/Users/vtyag/Downloads/archive/train-images.idx3-ubyte"
TRAIN_LABELS_PATH = r"C:/Users/vtyag/Downloads/archive/train-labels.idx1-ubyte"
TEST_IMAGES_PATH  = r"C:/Users/vtyag/Downloads/archive/t10k-images.idx3-ubyte"
TEST_LABELS_PATH  = r"C:/Users/vtyag/Downloads/archive/t10k-labels.idx1-ubyte"

CLASSES = (0, 1)
N_TRAIN_PER_CLASS = 1000
VAL_FRACTION = 0.2
PCA_COMPONENTS = 5
N_ROUNDS = 200
RANDOM_STATE = 42

def load_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    return data.reshape(num, rows * cols)

def load_idx_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)

class DecisionStump:
    def __init__(self):
        self.dim = None
        self.thresh = None
        self.polarity = 1
        self.error = None

    def fit(self, X, y, weights, n_cuts=3):
        n_samples, n_features = X.shape
        best_error = float('inf')
        for dim in range(n_features):
            x_min, x_max = X[:, dim].min(), X[:, dim].max()
            thresholds = np.linspace(x_min, x_max, n_cuts + 2)[1:-1]
            for thresh in thresholds:
                for polarity in (1, -1):
                    preds = np.ones(n_samples) * -polarity
                    mask = X[:, dim] <= thresh
                    preds[mask] = polarity
                    error = np.sum(weights[preds != y])
                    if error < best_error:
                        best_error = error
                        self.dim = dim
                        self.thresh = thresh
                        self.polarity = polarity
                        self.error = error
        return self

    def predict(self, X):
        preds = np.ones(X.shape[0]) * -self.polarity
        mask = X[:, self.dim] <= self.thresh
        preds[mask] = self.polarity
        return preds

class AdaBoost:
    def __init__(self, n_rounds=50):
        self.n_rounds = n_rounds
        self.stumps = []
        self.betas = []

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        n = X_train.shape[0]
        weights = np.full(n, 1/n)
        history = {
            'exp_loss_train': [], 
            'exp_loss_val': [], 
            'exp_loss_test': [], 
            'err_train': []
        }

        for t in range(self.n_rounds):
            stump = DecisionStump().fit(X_train, y_train, weights)
            err = stump.error
            beta = 0.5 * np.log((1 - err + 1e-10) / (err + 1e-10))

            self.stumps.append(stump)
            self.betas.append(beta)

            # Update weights
            pred_t = stump.predict(X_train)
            weights *= np.exp(-beta * y_train * pred_t)
            weights /= weights.sum()

            # Exponential loss
            F_train = self.decision_function(X_train)
            history['exp_loss_train'].append(np.mean(np.exp(-y_train * F_train)))

            # 0–1 training error via .predict()
            train_preds = self.predict(X_train)
            history['err_train'].append(np.mean(train_preds != y_train))

            if X_val is not None:
                F_val = self.decision_function(X_val)
                history['exp_loss_val'].append(np.mean(np.exp(-y_val * F_val)))
            if X_test is not None:
                F_test = self.decision_function(X_test)
                history['exp_loss_test'].append(np.mean(np.exp(-y_test * F_test)))

        # convert to arrays
        for k in history:
            history[k] = np.array(history[k])
        return history

    def decision_function(self, X):
        F = np.zeros(X.shape[0])
        for stump, beta in zip(self.stumps, self.betas):
            F += beta * stump.predict(X)
        return F

    def predict(self, X):
        # break ties in favor of +1, so never returns 0
        F = self.decision_function(X)
        return np.where(F >= 0, 1, -1)

def manual_pca(X, n_components):
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = np.dot(Xc.T, Xc) / (Xc.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    return mean, eigvecs[:, idx]

def load_data():
    X_train_full = load_idx_images(TRAIN_IMAGES_PATH)
    y_train_full = load_idx_labels(TRAIN_LABELS_PATH)
    X_test_full  = load_idx_images(TEST_IMAGES_PATH)
    y_test_full  = load_idx_labels(TEST_LABELS_PATH)
    
    mask_tr = np.isin(y_train_full, CLASSES)
    mask_te = np.isin(y_test_full, CLASSES)
    X_train_full, y_train_full = X_train_full[mask_tr], y_train_full[mask_tr]
    X_test, y_test = X_test_full[mask_te], y_test_full[mask_te]
    
    y_train_full = np.where(y_train_full == CLASSES[0], 1, -1)
    y_test = np.where(y_test == CLASSES[0], 1, -1)
    
    pos_idx = np.where(y_train_full == 1)[0][:N_TRAIN_PER_CLASS]
    neg_idx = np.where(y_train_full == -1)[0][:N_TRAIN_PER_CLASS]
    idx = np.hstack([pos_idx, neg_idx])
    X_sub, y_sub = X_train_full[idx], y_train_full[idx]
    
    mean_vec, comps = manual_pca(X_sub, PCA_COMPONENTS)
    X_sub_pca = (X_sub - mean_vec).dot(comps)
    X_test_pca = (X_test - mean_vec).dot(comps)
    
    np.random.seed(RANDOM_STATE)
    N = X_sub_pca.shape[0]
    perm = np.random.permutation(N)
    n_val = int(VAL_FRACTION * N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    
    X_val, y_val = X_sub_pca[val_idx], y_sub[val_idx]
    X_train, y_train = X_sub_pca[tr_idx], y_sub[tr_idx]
    
    return X_train, y_train, X_val, y_val, X_test_pca, y_test

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    model = AdaBoost(n_rounds=N_ROUNDS)
    history = model.fit(X_train, y_train, X_val, y_val, X_test, y_test)

    # use 1-based boosting rounds on x-axis
    rounds = np.arange(1, N_ROUNDS + 1)

    # Exponential loss plot
    plt.figure()
    plt.plot(rounds, history['exp_loss_train'], label='Train Exp-Loss')
    plt.plot(rounds, history['exp_loss_val'],   label='Val Exp-Loss')
    plt.plot(rounds, history['exp_loss_test'],  label='Test Exp-Loss')
    plt.xlabel('Boosting Round')
    plt.ylabel('Exponential Loss')
    plt.title('Exponential Loss vs Boosting Rounds')
    plt.xticks(np.arange(1, N_ROUNDS+1, 25))
    plt.grid(True)
    plt.legend()
    plt.show()

    # Training 0–1 error plot
    plt.figure()
    plt.plot(rounds, history['err_train'], label='Train 0–1 Error')
    plt.xlabel('Boosting Round')
    plt.ylabel('0–1 Error')
    plt.title('Training Error vs Boosting Rounds')
    plt.xticks(np.arange(1, N_ROUNDS+1, 25))
    plt.grid(True)
    plt.legend()
    plt.show()

    # Final evaluation
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Final Test Accuracy: {acc * 100:.2f}%")
    
    # First 20 labels
    true_labels = np.where(y_test == 1, CLASSES[0], CLASSES[1])
    pred_labels = np.where(y_pred == 1, CLASSES[0], CLASSES[1])
    print("\nFirst 20 True vs Predicted labels:")
    for i in range(20):
        print(f"Index {i}: True={true_labels[i]}, Pred={pred_labels[i]}")

    # Confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        i = 0 if t == CLASSES[0] else 1
        j = 0 if p == CLASSES[0] else 1
        cm[i, j] += 1
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
