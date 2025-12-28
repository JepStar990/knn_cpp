import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

plt.scatter(df.actual, df.predicted, alpha=0.5)
plt.plot([3,9],[3,9],'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("KNN Regression – Wine Quality")
plt.show()
