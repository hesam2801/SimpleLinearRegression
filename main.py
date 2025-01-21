import numpy as np
df = np.loadtxt("datasets/FuelConsumptionCo2.csv", delimiter=",", dtype=str)
eng_size = df[1:, 4]
cylinders = df[1:, 5]
fuel_city = df[1:, 8]
fuel_hwy = df[1:, 9]
fuel_comb = df[1:, 10]
fuel_comb_mpg = df[1:, 11]
co2 = df[1:, 12]


class LinearRegression:
    weights = []
    intercept = 0

    def __init__(self):
        self.weights = []
        self.intercept = 0

    def fit(self, X, y):
        y = y.astype('float64')
        ms = []
        for x in X:
            x = x.astype('float64')
            a = sum((x - np.mean(x)) * (y - np.mean(y))) / \
                sum((x - np.mean(x)) ** 2)
            self.weights.append(a)
            ms.append(a * np.mean(x))
        b = np.mean(y) - sum(ms)
        self.intercept = b
        print(l1.weights)
        print(l1.intercept)

    def predict(self, X):
        ms = []
        for i in range(len(self.weights)):
            x = X[i].astype('float64')
            ms.append(self.weights[i] * np.mean(x))
        y = sum(ms) + self.intercept
        print(y)


l1 = LinearRegression()
l1.fit(np.array([eng_size, fuel_comb]), co2)
l1.predict(np.array([eng_size, fuel_comb]))
