import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 데이터 입력
# -----------------------------
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2.1, 4.3, 5.9, 8.2, 9.8, 12.1, 14.2, 15.7])

# -----------------------------
# 선형회귀 (경사하강법)
# y = ax + b
# -----------------------------
a = 0
b = 0

learning_rate = 0.01
epochs = 5000
n = len(x)

for _ in range(epochs):

    y_pred = a * x + b

    da = (-2 / n) * np.sum(x * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    a -= learning_rate * da
    b -= learning_rate * db

linear_pred = a * x + b

print("===== 선형회귀 =====")
print(f"y = {a:.4f}x + {b:.4f}")

# -----------------------------
# 비선형회귀
# y = ax² + bx + c
# -----------------------------
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

params, _ = curve_fit(quadratic, x, y)

a2, b2, c2 = params

nonlinear_pred = quadratic(x, a2, b2, c2)

print("\n===== 비선형회귀 =====")
print(f"y = {a2:.4f}x² + {b2:.4f}x + {c2:.4f}")

# -----------------------------
# 오차 계산
# -----------------------------
linear_mse = np.mean((y - linear_pred) ** 2)
nonlinear_mse = np.mean((y - nonlinear_pred) ** 2)

print("\n선형회귀 MSE :", round(linear_mse, 4))
print("비선형회귀 MSE :", round(nonlinear_mse, 4))

# -----------------------------
# 그래프 출력
# -----------------------------
plt.figure(figsize=(8, 5))

plt.scatter(x, y, color="black", label="Data")
plt.plot(x, linear_pred, label="Linear Regression")
plt.plot(x, nonlinear_pred, label="Nonlinear Regression")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear vs Nonlinear Regression")
plt.legend()
plt.grid(True)

plt.show()