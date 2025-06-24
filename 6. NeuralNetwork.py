import random
import math
random.seed(42)

#1) Generate synthetic data
def generate_samples(mean, n):
    return [[random.gauss(mean[0], 1), random.gauss(mean[1], 1)] for _ in range(n)]

n_per_class = 10
samples0 = generate_samples([-1, -1], n_per_class)  
samples1 = generate_samples([ 1,  1], n_per_class)  

# Quick peek at first few samples
print("First 2 samples of class 0:", samples0[:2])
print("First 2 samples of class 1:", samples1[:2])
print()

train_data, train_labels = [], []
test_data,  test_labels  = [], []
for i in range(n_per_class):
    if i < n_per_class // 2:
        train_data.append(samples0[i]); train_labels.append(0)
        train_data.append(samples1[i]); train_labels.append(1)
    else:
        test_data.append(samples0[i]);  test_labels.append(0)
        test_data.append(samples1[i]);  test_labels.append(1)

print(f"Train size: {len(train_data)}   Test size: {len(test_data)}")
print()

#2) Initialize all parameters randomly
w1 = [random.uniform(-1, 1), random.uniform(-1, 1)]  # hidden weights
b1 = random.uniform(-1, 1)                           # hidden bias
w2 = random.uniform(-1, 1)                           # output weight
b2 = random.uniform(-1, 1)                           # output bias

print("Initial parameters:")
print(f"  w1 = {w1}")
print(f"  b1 = {b1:.4f}")
print(f"  w2 = {w2:.4f}")
print(f"  b2 = {b2:.4f}")
print()

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#3) Train with gradient descent + squared-error loss
lr     = 0.1
epochs = 1000

for epoch in range(1, epochs+1):
    for x, y in zip(train_data, train_labels):
        # Forward pass
        z1    = w1[0]*x[0] + w1[1]*x[1] + b1
        a1    = sigmoid(z1)
        z2    = w2*a1 + b2
        y_hat = z2

        # Backward pass
        dL_dy  = (y_hat - y)
        dw2    = dL_dy * a1
        db2    = dL_dy
        da1    = dL_dy * w2
        dz1    = da1 * a1 * (1 - a1)
        dw1_0  = dz1 * x[0]
        dw1_1  = dz1 * x[1]
        db1    = dz1

        # Updates
        w2    -= lr * dw2
        b2    -= lr * db2
        w1[0] -= lr * dw1_0
        w1[1] -= lr * dw1_1
        b1    -= lr * db1

    # Print training loss every 200 epochs
    if epoch % 200 == 0 or epoch == 1:
        train_mse = 0
        for tx, ty in zip(train_data, train_labels):
            ta1    = sigmoid(w1[0]*tx[0] + w1[1]*tx[1] + b1)
            ty_hat = w2*ta1 + b2
            train_mse += (ty_hat - ty)**2
        train_mse /= len(train_data)
        print(f"Epoch {epoch:4d}  Train MSE = {train_mse:.4f}")

print()
print("Parameters after training:")
print(f"  w1 = {w1}")
print(f"  b1 = {b1:.4f}")
print(f"  w2 = {w2:.4f}")
print(f"  b2 = {b2:.4f}")
print()

#4) Evaluate MSE on the test set
mse = 0.0
print("Test set predictions:")
for idx, (x, y) in enumerate(zip(test_data, test_labels), 1):
    a1    = sigmoid(w1[0]*x[0] + w1[1]*x[1] + b1)
    y_hat = w2*a1 + b2
    err   = (y_hat - y)**2
    mse  += err
    print(f"  Sample {idx:2d}:  x = {x},  true = {y},  pred = {y_hat:.4f},  sq_err = {err:.4f}")
mse /= len(test_data)

print()
print(f"Final Test MSE: {mse:.6f}")
