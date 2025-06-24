import numpy as np
import matplotlib.pyplot as plt


print("Generating synthetic dataset...")
np.random.seed(42)
x = np.random.rand(100)
y = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * x) + np.random.normal(0, 0.01, size=x.shape)

# Manual train/test split (80/20)
print("\nSplitting data into train/test (80/20)...")
indices = np.random.permutation(len(x))
split_idx = int(0.8 * len(x))
x_train, x_test = x[indices[:split_idx]], x[indices[split_idx:]]
y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")


cuts = np.linspace(0, 1, 22)[1:-1]


def fit_stump(x, target, cuts):
    """Find the best split among 'cuts' that minimizes squared error"""
    best_err = np.inf
    best_params = None

    for c in cuts:
        left_mask = x <= c
        right_mask = ~left_mask
        if not left_mask.any() or not right_mask.any():
            continue
        left_val = target[left_mask].mean()
        right_val = target[right_mask].mean()
        err = np.sum((target[left_mask] - left_val) ** 2) + np.sum((target[right_mask] - right_val) ** 2)
        if err < best_err:
            best_err = err
            best_params = (c, left_val, right_val)

    return best_params

class DecisionStump:
    def __init__(self, split, left_val, right_val):
        self.split = split
        self.left_val = left_val
        self.right_val = right_val

    def predict(self, x):
        return np.where(x <= self.split, self.left_val, self.right_val)


def gradient_boosting(x_train, y_train, x_test, y_test,
                      loss='squared', n_estimators=100, lr=0.01):
    print(f"\nStarting Gradient Boosting with {loss} loss...")
    print(f"Initial prediction: {y_train.mean():.3f}")
    
    # Initialize prediction as mean of y_train
    F_train = np.full_like(y_train, y_train.mean(), dtype=float)
    F_test = np.full_like(y_test, y_train.mean(), dtype=float)

    train_losses = []
    preds_train = [F_train.copy()]
    preds_test = [F_test.copy()]
    learners = []

    for m in range(1, n_estimators + 1):
        # Compute negative gradient
        if loss == 'squared':
            residual = y_train - F_train
        elif loss == 'absolute':
            residual = np.sign(y_train - F_train)
        
        # Fit stump to residuals
        split, left_val, right_val = fit_stump(x_train, residual, cuts)
        stump = DecisionStump(split, left_val, right_val)
        learners.append(stump)

        # Update predictions with learning rate
        F_train += lr * stump.predict(x_train)
        F_test += lr * stump.predict(x_test)

        # Record predictions and loss
        preds_train.append(F_train.copy())
        preds_test.append(F_test.copy())
        loss_val = np.mean((y_train - F_train)**2) if loss == 'squared' else np.mean(np.abs(y_train - F_train))
        train_losses.append(loss_val)

        # Progress output
        if m % 10 == 0 or m == 1:
            print(f"Iter {m:3d}/{n_estimators} | Train Loss: {loss_val:.4f}")

    print(f"{loss.capitalize()} loss training completed!")
    return learners, preds_train, preds_test, train_losses


n_estimators = 100
learning_rate = 0.01

print("\n" + "="*50)
print("Training with Squared Loss:")
print("="*50)
res_sq = gradient_boosting(x_train, y_train, x_test, y_test,
                           loss='squared', n_estimators=n_estimators, lr=learning_rate)

print("\n" + "="*50)
print("Training with Absolute Loss:")
print("="*50)
res_abs = gradient_boosting(x_train, y_train, x_test, y_test,
                            loss='absolute', n_estimators=n_estimators, lr=learning_rate)


print("\nGenerating training loss plot...")
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_estimators + 1), res_sq[3], label='Squared Loss')
plt.plot(range(1, n_estimators + 1), res_abs[3], label='Absolute Loss')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss over Iterations')
plt.legend()
plt.tight_layout()
plt.show()


print("\nGenerating prediction plots...")
iters_to_plot = [0, 10, 50, 100]

for loss_name, (preds_tr, preds_te) in zip(
    ['Squared', 'Absolute'], 
    [(res_sq[1], res_sq[2]), (res_abs[1], res_abs[2])]
):
    print(f"\nCreating {loss_name} Loss plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Plot training predictions
    for i in iters_to_plot:
        axes[0].scatter(x_train, preds_tr[i], s=15, alpha=0.6, label=f'Iter {i}')
    axes[0].scatter(x_train, y_train, color='k', s=10, label='True')
    axes[0].set_title(f'Train ({loss_name} Loss)')
    axes[0].set_xlabel('x')
    axes[0].legend()
    
    # Plot test predictions
    for i in iters_to_plot:
        axes[1].scatter(x_test, preds_te[i], s=15, alpha=0.6, label=f'Iter {i}')
    axes[1].scatter(x_test, y_test, color='k', s=10, label='True')
    axes[1].set_title(f'Test ({loss_name} Loss)')
    axes[1].set_xlabel('x')
    axes[1].legend()
    
    plt.suptitle(f'Gradient Boosting: {loss_name} Loss')
    plt.tight_layout()
    plt.show()

print("\nAll plots generated successfully!")