#!/usr/bin/env python
# coding: utf-8

# **Prompt:**
# Write code in python to generate the data for X an y where y = 3x - 4 + random noise You can use numpy.

# https://chatgpt.com/share/68c4e8ba-9240-8010-84f1-3567f4031f2f

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate 100 data points for X
X = np.linspace(0, 10, 100)

# Generate random noise (normal distribution with mean 0 and std deviation 1)
noise = np.random.normal(0, 1, X.shape)

# Generate y values
y = 3 * X - 4 + noise

# Plot to visualize
plt.scatter(X, y, label="Data with noise")
plt.plot(X, 3*X - 4, color="red", label="True line (no noise)")
plt.legend()
plt.show()


# Create a function predict that takes X and weights w and gives y by y = w.X. add column of 1s for bias before doing w.X

# In[ ]:


def predict(X, w):
    """
    Predicts y = w.X after adding a bias column of 1s to X.

    Parameters:
        X : numpy array of shape (n_samples, n_features)
            Input features
        w : numpy array of shape (n_features + 1,)
            Weights (including bias as last element)

    Returns:
        y : numpy array of shape (n_samples,)
            Predicted values
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Add column of 1s for bias
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

    # Compute y = X_bias @ w
    y = X_bias @ w
    return y


# Example usage:
np.random.seed(42)
X = np.linspace(0, 10, 5)  # 5 points
w = np.array([3, -4])      # slope=3, bias=-4

y_pred = predict(X, w)
print("X:\n", X)
print("Predicted y:\n", y_pred)


# Now, define an error function which takes the following arguments: training Data X, training labels y, weights W, loss function loss. It should first use our predict function to calculate ppred and then calculate the overall loss by using the loss on ypred, y

# In[ ]:


import numpy as np

def error(X, y, w, loss):
    """
    Compute error using a given loss function.

    Parameters:
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Training labels
        w : numpy array of shape (n_features + 1,)
            Weights (including bias)
        loss : function
            A function that takes (y_true, y_pred) and returns scalar loss

    Returns:
        total_loss : float
            Overall loss value
    """
    # Predict using our linear model
    ypred = predict(X, w)

    # Compute and return loss
    return loss(y, ypred)


# Example loss functions
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# In[ ]:


# Example usage
np.random.seed(42)
X = np.linspace(0, 10, 20)
w = np.array([4, -4])   # slope=3, bias=-4
y = 3 * X - 4 + np.random.normal(0, 1, X.shape)  # training labels with noise

print("MSE:", error(X, y, w, mse))
print("MAE:", error(X, y, w, mae))


# In[ ]:


error(X, y, [2, 3], mse)


# Write a function 'gradient' to compute the gradient of a loss function 'loss' wrt to each weight at given X, y, weights.

# In[ ]:


import numpy as np

def gradient(X, y, w, loss, error=error,eps=1e-6):
    """
    Compute numerical gradient of loss wrt weights.

    Parameters:
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Training labels
        w : numpy array of shape (n_features + 1,)
            Weights (including bias)
        loss : function
            A function that takes (y_true, y_pred) and returns scalar loss
        eps : float
            Small epsilon for numerical approximation

    Returns:
        grad : numpy array of shape (n_features + 1,)
            Gradient of loss wrt each weight
    """
    grad = np.zeros_like(w, dtype=float)

    # Current loss
    base_loss = error(X, y, w, loss)

    # Finite difference approximation
    for i in range(len(w)):
        w_perturbed = w.copy()
        w_perturbed[i] += eps
        loss_perturbed = error(X, y, w_perturbed, loss)
        grad[i] = (loss_perturbed - base_loss) / eps

    return grad


# Example usage
np.random.seed(42)
X = np.linspace(0, 10, 20)
y = 3 * X - 4 + np.random.normal(0, 1, X.shape)
w = np.array([0.0, 0.0])   # initial guess

g = gradient(X, y, w, mse)
print("Gradient at w =", w, "is", g)


# Define a function learners that takes X, y, error(), predict(), loss(). It assumes somes values of W. Computes the gradiants using above function and then uses gradient descent to optimize the values of weights.  and it returns the weights.

# In[ ]:


ww = np.array([1., 2.])
gg = np.array([10., -15.])
eta = .001
ww -= eta * gg
ww


# In[ ]:


import numpy as np
# assuming predict(), error(), loss (e.g., mse), and gradient() are already defined

def learners(X, y, error_fn, predict_fn, loss_fn, lr=0.01, epochs=1000, tol=1e-8, w_init=None, verbose=False):
    """
    Optimize weights with gradient descent and return the learned weights.

    Parameters:
        X : (n_samples, n_features) or (n_samples,)
        y : (n_samples,)
        error_fn : function(X, y, w, loss_fn) -> float
            Overall loss calculator (uses predict_fn internally if your version supports it).
        predict_fn : function(X, w) -> y_pred
            Not used directly here (error_fn uses it), included for API symmetry as requested.
        loss_fn : function(y_true, y_pred) -> float
        lr : float
            Learning rate.
        epochs : int
            Max iterations.
        tol : float
            Stop if ||grad||_2 < tol.
        w_init : np.ndarray or None
            Initial weights (length = n_features + 1). If None, initialized to zeros.
        verbose : bool
            If True, prints progress.

    Returns:
        w : np.ndarray of shape (n_features + 1,)
            Learned weights (including bias as last term).
    """
    # Ensure shapes
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    w = np.zeros(n_features + 1) if w_init is None else np.asarray(w_init, dtype=float)

    for t in range(epochs):
        g = gradient(X, y, w, loss_fn, error_fn)  # numerical gradient wrt each weight
        w = w - lr * g

        # Convergence check
        g_norm = np.linalg.norm(g, ord=2)
        if verbose and (t % max(1, epochs // 10) == 0 or g_norm < tol):
            curr_loss = error_fn(X, y, w, loss_fn)
            print(f"iter {t:4d} | loss={curr_loss:.6f} | ||grad||={g_norm:.3e}")
            print(w)

        if g_norm < tol:
            break

    return w


# ---- Example usage ----
# Define a simple MSE if you don't already have one
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Synthetic data
# np.random.seed(42)
# X_demo = np.linspace(0, 10, 50)
# noise = np.random.normal(0, 1, X_demo.shape)
# y_demo = 3 * X_demo - 4 + noise

X_demo, y_demo = X, y
# Learn weights starting from zeros
w_learned = learners(
    X_demo, y_demo,
    error_fn=error,          # uses your previously defined error()
    predict_fn=predict,      # included for API symmetry
    loss_fn=mse,
    lr=0.01,
    epochs=2000,
    tol=1e-8,
    w_init=None,
    verbose=True
)

print("Learned weights [slope, bias]:", w_learned)


# Plot X, y along with a line represented by w.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def plot_xy_with_line(X, y, w, title="Data & Fitted Line"):
    """
    Plots X vs y as scatter and overlays the line represented by weights w.
    Assumes a single feature: y = w[0]*x + w[1].

    Parameters:
        X : array-like, shape (n_samples,) or (n_samples, 1)
        y : array-like, shape (n_samples,)
        w : array-like, shape (2,) -> [slope, bias]
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # Ensure single feature
    if X.ndim == 2:
        if X.shape[1] != 1:
            raise ValueError("plot_xy_with_line supports only 1 feature (X shape must be (n, ) or (n, 1)).")
        x1d = X.ravel()
    else:
        x1d = X

    # Build a smooth line over the X range
    x_line = np.linspace(x1d.min(), x1d.max(), 200)
    y_line = w[0] * x_line + w[1]

    plt.figure()
    plt.scatter(x1d, y, label="Data")
    plt.plot(x_line, y_line, label=f"Line: y = {w[0]:.3f}x + {w[1]:.3f}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



# In[ ]:


plot_xy_with_line(X_demo, y_demo, w_learned)

# Or with known weights:
# w_true = np.array([3.0, -4.0])
# plot_xy_with_line(X_demo, y_demo, w_true)


# Now, generate a dataset for classification, say X is age of people and y represents is_adult. The people above 18 are considered adult 

# In[ ]:


import numpy as np

def make_is_adult(n=200, min_age=0, max_age=80, seed=42, integer=True, flip_rate=0.0):
    """
    Generate a classification dataset: X = age, y = is_adult (1 if age > 18 else 0).

    Parameters:
        n         : number of samples
        min_age   : minimum possible age (inclusive)
        max_age   : maximum possible age (inclusive if integer=True)
        seed      : RNG seed for reproducibility
        integer   : if True, ages are integers; else float ages
        flip_rate : fraction of labels to randomly flip (simulate noise)

    Returns:
        X : np.ndarray shape (n,) of ages
        y : np.ndarray shape (n,) of 0/1 labels
    """
    rng = np.random.default_rng(seed)
    if integer:
        X = rng.integers(min_age, max_age + 1, size=n)
    else:
        X = rng.uniform(min_age, max_age, size=n)

    y = (X > 18).astype(int)  # strictly "above 18" => adult

    if flip_rate > 0:
        flip_mask = rng.random(n) < flip_rate
        y[flip_mask] = 1 - y[flip_mask]

    return X, y


# Example usage
if __name__ == "__main__":
    X, y = make_is_adult(n=20, flip_rate=0.0)
    print("Ages (X):", X)
    print("is_adult (y):", y)
    print("Adults count:", y.sum(), "/", len(y))


# Write code to generate the data using this function and plot it. Y axis is for is_adult or not. X_axis the age

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# --- assumes make_is_adult() from earlier is already defined ---
# from your previous cell:
# def make_is_adult(n=200, min_age=0, max_age=80, seed=42, integer=True, flip_rate=0.0): ...

# Generate data
X, y = make_is_adult(n=300, min_age=0, max_age=80, seed=42, integer=True, flip_rate=0.0)

# OPTIONAL: jitter y slightly so overlapping points are easier to see
y_plot = y + np.random.uniform(-0.04, 0.04, size=y.shape)

# Plot
plt.figure(figsize=(8, 4))
plt.scatter(X, y_plot, alpha=0.7, edgecolor="none")
plt.axvline(18, linestyle="--", label="threshold: age > 18")

plt.yticks([0, 1], ["not adult (0)", "adult (1)"])
plt.xlabel("Age")
plt.ylabel("is_adult")
plt.title("Is Adult vs Age")
plt.legend()
plt.grid(axis="x", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()


# Use the learners() function to learn from this using the mse and plot the result.

# In[ ]:


X, y = make_is_adult(n=300, min_age=0, max_age=80, seed=42, integer=True, flip_rate=0.0)

# Learn linear model with MSE
w_learned = learners(
    X, y,
    error_fn=error,
    predict_fn=predict,
    loss_fn=mse,
    lr=1e-4,        # small LR for stability with numerical gradients
    epochs=5000,
    tol=1e-8,
    w_init=None,
    verbose=False
)
print("Learned weights [slope, bias]:", w_learned)

# Plot result
plot_xy_with_line(X, y, w_learned)


# Create logloss function for binary classification loss. Now, using the 'learners' function defined above and using logloss function learn the weights from X, y (from is_adult dataset)

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# --- Assume these already exist from earlier: make_is_adult, predict, error, gradient, learners ---

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logloss(y_true, y_linear):
    """
    Binary cross-entropy. Expects *linear* predictions; applies sigmoid inside.
    y_true in {0,1}
    """
    y_true = np.asarray(y_true).astype(float)
    p = sigmoid(np.asarray(y_linear))
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


# --- Generate data ---
X, y = make_is_adult(n=300, min_age=0, max_age=80, seed=42, integer=True, flip_rate=0.0)

# --- Learn weights with logloss (logistic regression via numerical gradient) ---
w_log = learners(
    X, y,
    error_fn=error,         # uses predict() -> linear output
    predict_fn=predict,     # included for API symmetry
    loss_fn=logloss,        # applies sigmoid internally
    lr=1e-2,                # tune if needed
    epochs=6000,
    tol=1e-8,
    w_init=None,
    verbose=False
)

print("Learned weights (logistic): [slope, bias] =", w_log)

# --- Optional: visualize probability curve and decision boundary ---
ages = np.linspace(X.min(), X.max(), 400)
probs = sigmoid(predict(ages.reshape(-1, 1), w_log))
x0 = -w_log[1] / w_log[0]  # 0.5-probability threshold

# Jitter the y for visibility (0/1 points)
y_jitter = y + np.random.uniform(-0.04, 0.04, size=y.shape)

plt.figure(figsize=(8, 4))
plt.scatter(X, y_jitter, alpha=0.6, edgecolor="none", label="Data (0/1 with jitter)")
plt.plot(ages, probs, label="Predicted P(is_adult|age)", linewidth=2)
plt.axhline(0.5, linestyle="--", linewidth=1, label="P=0.5")
plt.axvline(x0, linestyle="--", linewidth=1, label=f"Decision boundary ≈ {x0:.2f}")
plt.axvline(18, linestyle=":", linewidth=1, label="True threshold = 18")
plt.yticks([0, 0.5, 1], ["0", "0.5", "1"])
plt.xlabel("Age")
plt.ylabel("Probability of is_adult")
plt.title("Logistic fit learned via learners() + logloss")
plt.legend()
plt.tight_layout()
plt.show()


# Generate the data from a slab based calculation like tax. if the salary is less than 1000, the tax is 0 other wise tax rate is 30%

# In[ ]:


#Here’s a tiny, vectorized NumPy helper to generate a **slab-based tax dataset**.

#Assumption: the 30% tax applies **only to the amount above the threshold** (classic slab). You can flip a flag to apply 30% to the whole salary instead.

import numpy as np
import matplotlib.pyplot as plt

def make_tax_dataset(
    n=300,
    min_salary=0,
    max_salary=10000,
    threshold=1000,
    rate=0.30,
    apply_on_excess=True,   # True: tax on (salary - threshold); False: tax on full salary once threshold crossed
    integer=True,
    seed=42
):
    """
    Generate (X, y) where:
      X = salary
      y = tax payable under a simple slab rule:
          - if salary < threshold: tax = 0
          - else:
              apply_on_excess=True  -> tax = rate * (salary - threshold)
              apply_on_excess=False -> tax = rate * salary
    """
    rng = np.random.default_rng(seed)
    if integer:
        X = rng.integers(min_salary, max_salary + 1, size=n)
    else:
        X = rng.uniform(min_salary, max_salary, size=n)

    if apply_on_excess:
        y = rate * np.clip(X - threshold, a_min=0, a_max=None)
    else:
        y = np.where(X < threshold, 0.0, rate * X)

    return X, y

# ---- Example usage + plot ----
X, y = make_tax_dataset(n=300, min_salary=0, max_salary=10000, threshold=1000, rate=0.30, apply_on_excess=True)

plt.figure(figsize=(8, 4))
plt.scatter(X, y, s=12, alpha=0.8, edgecolor="none", label="(salary, tax)")
plt.axvline(1000, linestyle="--", label="Threshold = 1000")
plt.xlabel("Salary")
plt.ylabel("Tax")
plt.title("Slab-based Tax Dataset (30% on amount above threshold)")
plt.legend()
plt.tight_layout()
plt.show()

# * `X` is the salary array; `y` is the corresponding tax.
# * Set `apply_on_excess=False` if you want 30% on the **entire** salary once it crosses the threshold.


# write code to train a regressor using mse in learners() on this X and y and show draw the line on the plot along with X, y

# In[ ]:


X, y = make_tax_dataset(n=300, min_salary=0, max_salary=10000,
                        threshold=1000, rate=0.30, apply_on_excess=True)

# Learn linear regressor with MSE
w_learned = learners(
    X, y,
    error_fn=error,
    predict_fn=predict,
    loss_fn=mse,
    lr=1e-8,         # small LR because salaries/taxes are large-magnitude
    epochs=20000,
    tol=1e-8,
    w_init=None,
    verbose=False
)
print("Learned weights [slope, bias]:", w_learned)

# Plot data and fitted line
plot_xy_with_line(X, y, w_learned)


# come up with a new label y_is_taxable and train a classifier using learners() method on X, and y_is_taxable

# In[ ]:


threshold = 1000
y_is_taxable = (y > 0).astype(int)

# Scale feature for stable/fast optimization (keeps learners() unchanged)
scale = 1000.0
X_scaled = X / scale

w_cls = learners(
    X_scaled, y_is_taxable,
    error_fn=error,
    predict_fn=predict,
    loss_fn=logloss,
    lr=2e-2,        # works well after scaling
    epochs=6000,
    tol=1e-8,
    w_init=None,
    verbose=False
)


# In[ ]:


print("Learned weights (on scaled X): [slope, bias] =", w_cls)

# Decision boundary (P=0.5): w0 * x_scaled + b = 0  =>  x_scaled = -b/w0  =>  x = scale * x_scaled
x0_scaled = -w_cls[1] / (w_cls[0] + 1e-12)
x0 = x0_scaled * scale
print(f"Estimated decision boundary ≈ {x0:.2f} (true threshold = {threshold})")

# ---------- Visualize ----------
xs = np.linspace(X.min(), X.max(), 500)
ps = sigmoid(predict((xs/scale).reshape(-1, 1), w_cls))

# Jitter labels for visibility
y_jitter = y_is_taxable + np.random.uniform(-0.04, 0.04, size=y_is_taxable.shape)

plt.figure(figsize=(8, 4))
plt.scatter(X, y_jitter, alpha=0.6, edgecolor="none", s=14, label="y_is_taxable (0/1, jittered)")
plt.plot(xs, ps, linewidth=2, label="Predicted P(taxable | salary)")
plt.axhline(0.5, linestyle="--", linewidth=1, label="P=0.5")
plt.axvline(x0, linestyle="--", linewidth=1, label=f"Decision boundary ≈ {x0:.0f}")
plt.axvline(threshold, linestyle=":", linewidth=1, label=f"True threshold = {threshold}")
plt.yticks([0, 0.5, 1], ["0", "0.5", "1"])
plt.xlabel("Salary")
plt.ylabel("Probability of being taxable")
plt.title("Classifier trained with learners() + logloss")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


y


# In[ ]:


scores = predict(X_scaled.reshape(-1, 1), w_cls)
is_taxable_hat = (sigmoid(scores) >= 0.5).astype(float)

# 3) Create X1 = [salary, is_taxable_hat] and train a regressor with MSE
X1 = np.column_stack([X, is_taxable_hat])
w_reg = learners(
    X1, y,
    error_fn=error,
    predict_fn=predict,
    loss_fn=mse,
    lr=0.000001, epochs=30000, tol=1e-8, verbose=False
)
print("Regressor weights [w_salary, w_is_taxable, bias]:", w_reg)

# 4) Plot data and fitted line (use classifier to compute is_taxable feature along the x-grid)
x_line = np.linspace(X.min(), X.max(), 500)
is_taxable_line = (sigmoid(predict((x_line/scale).reshape(-1,1), w_cls)) >= 0.5).astype(float)
X1_line = np.column_stack([x_line, is_taxable_line])
y_line = predict(X1_line, w_reg)

plt.figure(figsize=(8, 4))
plt.scatter(X, y, s=12, alpha=0.8, edgecolor="none", label="(salary, tax)")
plt.plot(x_line, y_line, linewidth=2, label="Regressor fit using feature: is_taxable")
plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold}")
plt.xlabel("Salary")
plt.ylabel("Tax")
plt.title("Tax vs Salary: Linear Regressor with learned 'is_taxable' feature")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# ![image.png](attachment:image.png)

# In[ ]:


X = np.array([1,2,3,4])
print(X)
X = X.reshape(-1,1)
X


# In[ ]:


X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
X_bias


# In[ ]:


w = np.array([1,2,3, 4])
w1 = w[:2]
w2 = w[2:]
w1, w2


# In[ ]:


X = np.array([[1000], [2000], [3000]])

print(y)


# In[ ]:


y


# In[ ]:


def relu(x):
    """
    Apply ReLU activation: f(x) = max(0, x)

    Parameters:
        x : np.ndarray or scalar
            Input values.

    Returns:
        np.ndarray or scalar
            Output after applying ReLU.
    """
    return np.maximum(0, x)


# In[ ]:


def predict_nn(X, w):
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    w1 = w[:2]
    w2 = w[2:]

    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    y1 = relu(X_bias @ w1)
    print("y1", y1)
    Y1 = y1.reshape(-1, 1)
    Y1_bias = np.hstack([Y1, np.ones((Y1.shape[0], 1))])
    ypred = Y1_bias @ w2
    print("ypred", ypred)
    return ypred


# In[ ]:


def error_nn(X, y, w, loss):
    ypred = predict_nn(X, w)
    return loss(y, ypred)


# In[ ]:


X, y = make_tax_dataset(n=300, min_salary=0, max_salary=10000,
                        threshold=1000, rate=0.30, apply_on_excess=True)

w_learned = np.array([.1,.2,.3,.4])



# In[ ]:


# Learn linear regressor with MSE
w_learned = learners(
    X, y,
    error_fn=error_nn,
    predict_fn=predict_nn,
    loss_fn=mse,
    lr=1e-5,         # small LR because salaries/taxes are large-magnitude
    epochs=20000,
    tol=1e-8,
    w_init=w_learned,
    verbose=False
)
print("Learned weights [slope, bias]:", w_learned)


# In[ ]:


X


# In[ ]:


w_learned


# In[ ]:


ypred = predict_nn(X, w_learned)
ypred, y


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Generate input values
X = np.linspace(-10, 10, 400)
Y = relu(X)

# Plot
plt.figure(figsize=(6,4))
plt.plot(X, Y, label="ReLU(x)", linewidth=2)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# In[ ]:


o = np.array([10,20,30])
eo1 = np.exp(o)
eo1/np.sum(eo1)


# In[ ]:


o = np.array([1,2,3,5, 1,])
eo1 = np.exp(o)
eo1/np.sum(eo1)


# In[ ]:




