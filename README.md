# ML Algorithms From Scratch

A collection of classic machine learning algorithms and experiments implemented from the ground up in Python. This repository demonstrates both supervised and unsupervised methods, explores loss functions, ensemble techniques, and simple neural networks—all written from scratch to deepen understanding of underlying mechanics.

## Table of Contents

1. [Digit Detection (PCA, FDA, Discriminant Analysis)](#digit-detection-pca-fda-discriminant-analysis)
2. [Decision Tree, Random Forest & Bagging](#decision-tree-random-forest--bagging)
3. [Polynomial Regression & Sine Approximation](#polynomial-regression--sine-approximation)
4. [AdaBoost With Decision Stumps](#adaboost-with-decision-stumps)
5. [Gradient Boosting for Regression](#gradient-boosting-for-regression)
6. [Neural Network for Binary Classification](#neural-network-for-binary-classification)

---

## Getting Started

### Prerequisites

* Python 3.7+
* NumPy
* SciPy
* scikit-learn (for dataset utilities and performance comparison)
* Matplotlib (for plots)

Install dependencies:

```bash
pip install numpy scipy scikit-learn matplotlib
```

### Repository Structure

```text
├── Digit_Detection.py
├── Decision_Tree_RandomForest.ipynb
├── Regression.ipynb
├── AdaBoost.py
├── GradientBoosting.py
└── NeuralNetwork.py
```

Each script/notebook corresponds to one algorithm or experiment, with detailed comments and output visualizations.

---

## Digit Detection (PCA, FDA, Discriminant Analysis)

**File:** `Digit_Detection.py`
Reduces dimensionality of MNIST digits via PCA and Fisher Discriminant Analysis (FDA), then applies linear and quadratic discriminant classifiers to identify handwritten digits. Includes accuracy, confusion matrices, and runtime comparisons to highlight pros and cons of each projection method.

**Usage:**

```bash
python Digit_Detection.py
```

---

## Decision Tree, Random Forest & Bagging

**File:** `Decision_Tree_RandomForest.ipynb`
Implements a binary decision tree using the Gini impurity criterion, bootstrap aggregating (bagging), and a random forest from scratch. Compares single-tree, bagged, and forest ensembles on benchmark datasets to illustrate variance reduction and accuracy gains.

**Usage:**

* Open the notebook in Jupyter:

  ```bash
  jupyter notebook Decision_Tree_RandomForest.ipynb
  ```

---

## Polynomial Regression & Sine Approximation

**File:** `Regression.ipynb`
Experiments to find the optimal polynomial degree for approximating the sine function. Fits both Taylor-series and least-squares polynomials over a specified interval, then compares mean squared and maximum errors across degrees to select the best trade-off.

**Usage:**

* Open the notebook:

  ```bash
  jupyter notebook Regression.ipynb
  ```

---

## AdaBoost With Decision Stumps

**File:** `AdaBoost.py`
A pure-Python AdaBoost implementation using decision stumps as weak learners on PCA-reduced MNIST digits (classifying 0 vs 1). Tracks weight updates, training error per boosting round, and contrasts ensemble accuracy with a single stump baseline.

**Usage:**

```bash
python AdaBoost.py
```

---

## Gradient Boosting for Regression

**File:** `GradientBoosting.py`
Scratch-built gradient boosting regressor that sequentially fits decision stumps to residuals. Supports both squared-error and absolute-error loss functions, and includes comparative plots of convergence and predictive performance to showcase robustness differences.

**Usage:**

```bash
python GradientBoosting.py
```

---

## Neural Network for Binary Classification

**File:** `NeuralNetwork.py`
A minimal feed-forward neural network for binary classification with two input features, one hidden neuron (sigmoid activation), and a linear output neuron. Trains on synthetic data using backpropagation, with loss-curve tracking and decision boundary visualization.


