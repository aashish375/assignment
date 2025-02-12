import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Exercise 1: Regression to the Mean
n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    s = dice1 + dice2
    
    h, h2 = np.histogram(s, bins=range(2, 14))
    
    plt.bar(h2[:-1], h / n)
    plt.title(f"Dice Sum Distribution for n={n}")
    plt.xlabel("Sum of Dice")
    plt.ylabel("Frequency")
    plt.show()