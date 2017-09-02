import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the data from CSV
input_file = "notebook/data/ground-cricket-chirps.csv"
df = pd.read_csv(input_file, header=0)
df.columns = ['chirps','temp']

# Let's take a look at the data
print(df.head(10))

x = df['temp']
y = df['chirps']

# Find the linear regression equation for this data
X = x.to_frame()
reg = LinearRegression()
reg.fit(X, y)
pred = reg.predict(X)
# The coefficients
print('Coefficients: ', reg.coef_)
# The intercept
print('Intercept: ', reg.intercept_)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y, pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, pred))

# Plot the data
plt.scatter(X, y)
plt.plot(X, (reg.intercept_ + reg.coef_ * X))
plt.xlabel("Chirps/sec")
plt.ylabel("Ground Temperature")
plt.show()


