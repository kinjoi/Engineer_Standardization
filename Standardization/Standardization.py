
import numpy as np

class Standardization:
    def fit_tranfsorm(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        Xsd = (X - self.mean_) / self.std_
        return Xsd
    
    def transform(self, X):
        Xsd = (X - self.mean_) / self.std_
        return Xsd

    def inverse_transform(self, Xsd):
        X = Xsd * self.std_ + self.mean_
        return X

X_train = np.array([[1, 100], [2, 200], [3, 300]])

stand = Standardization()
Xsd_train = stand.fit_tranfsorm(X_train)
print("Xsd = ", X_train)
print("Xsd_train = ", Xsd_train)
print("Xsd (inversed) = ", stand.inverse_transform(Xsd_train))

X_test = np.array([[-2, 198], [0, 199], [2, 200], [+4, +201], [+6, +202]])

print("X_test = ", stand.transform(X_test))

