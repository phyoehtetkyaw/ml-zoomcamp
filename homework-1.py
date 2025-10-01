import pandas as pd
import numpy as np

'''
Q1. Pandas version
What version of Pandas did you install?

You can get the version information using the __version__ field:
'''
print(pd.__version__)

data = pd.read_csv('data/car_efficiency_fuel.csv')
df = pd.DataFrame(data)

print(df)

'''
Q2. Records count
How many records are in the dataset?

4704
8704
9704
17704
'''
print(len(df))

'''
Q3. Fuel types
How many fuel types are presented in the dataset?

1
2
3
4
'''
print(df["fuel_type"].value_counts())

'''
Q4. Missing values
How many columns in the dataset have missing values?

0
1
2
3
4
'''
print(df.isnull().sum())

'''
Q5. Max fuel efficiency
What's the maximum fuel efficiency of cars from Asia?

13.75
23.75
33.75
43.75
'''
print(df.loc[df["origin"] == "Asia", "fuel_efficiency_mpg"].max())

print("=" * 100)

'''
Q6. Median value of horsepower
Find the median value of the horsepower column in the dataset.
Next, calculate the most frequent value of the same horsepower column.
Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
Now, calculate the median value of horsepower once again.
Has it changed?

Yes, it increased
Yes, it decreased
No
'''

print("Before Mean horsepower: ", df["horsepower"].mean())
print(df["horsepower"].value_counts())
df["horsepower"] = df["horsepower"].fillna(152)
print("After Mean horsepower: ", df["horsepower"].mean())

print("=" * 100)

'''
Q7. Sum of weights
Select all the cars from Asia
Select only columns vehicle_weight and model_year
Select the first 7 values
Get the underlying NumPy array. Let's call it X.
Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
Invert XTX.
Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
What's the sum of all the elements of the result?
Note: You just implemented linear regression. We'll talk about it in the next lesson.

0.051
0.51
5.1
51
'''

X = np.array(df.loc[df["origin"] == "Asia", ["vehicle_weight", "model_year"]][:7])
TX = X.T
XTX = np.dot(TX, X)
IXTX = np.linalg.inv(XTX)
Y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
IXTXTX = np.dot(IXTX, TX)
W = np.dot(IXTXTX, Y)
print(W.sum())