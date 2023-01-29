import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
from sklearn.datasets import load_iris



iris = load_iris()

iris_df = pd.DataFrame(data= np.c_[iris["data"], iris["target"]], columns= iris["feature_names"] + ["target"])

classesiris_df = iris_df[iris_df["target"].isin([0,1])]


## replace every 0 by -1
iris_df["target"] = iris_df[["target"]].replace(0,-1)

iris_df = iris_df[["petal length (cm)", "petal width (cm)", "target"]]
iris_df.head()

X = iris_df[["petal length (cm)", "petal width (cm)"]].to_numpy()

y = iris_df[["target"]].to_numpy()


plt.figure(figsize=(8, 8))
colors = ["steelblue", "orange"]
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), alpha=0.5, cmap=matplotlib.colors.ListedColormap(colors), edgecolors="black")


plt.show()

n = X.shape[0]


##  H = sum(y[i] *y[j]*x[i]*x[j])
H = np.dot(y*X, (y*X).T)

## -1 * objective to convert to minimze
q = np.repeat([-1.0], n)[..., None]
## vector contains y for the first constriant
A = y.reshape(1, -1)
##bias
b = 0.0

## matrix n*n with -1 at diagonal for second constraint
G = np.negative(np.eye(n))
## vector of zeros for alphas
h = np.zeros(n)


P = matrix(H)
q = matrix(q)
G = matrix(G)
h = matrix(h)
A = matrix(A)
b = matrix(b)
sol = solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol["x"])


print(alphas)
## w = sum(alpha * y * x)
w = np.dot((y * alphas).T, X)[0]

S = (alphas >= 0.0000001).flatten()

## bias = 1/y - w_transpose * x
b = np.mean(y[S] - np.dot(X[S], w.reshape(-1,1)))

print("W:", w)
print("b:", b)

x_min = 0
x_max = 5.5
y_min = 0
y_max = 2

xx = np.linspace(x_min, x_max)


a = -w[0]/w[1]
yy = a*xx - (b)/w[1]
#yy = xx.dot(w.reshape(xx.s)) + (b)
##

margin = 1 / np.sqrt(np.sum(w**2))
yy_neg = yy - np.sqrt(1 + a**2) * margin
yy_pos = yy + np.sqrt(1 + a**2) * margin
plt.figure(figsize=(8, 8))
plt.plot(xx, yy, "b-")
plt.plot(xx, yy_neg, "m--")
plt.plot(xx, yy_pos, "m--")
colors = ["steelblue", "orange"]
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), alpha=0.5, cmap=matplotlib.colors.ListedColormap(colors), edgecolors="black")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
