import numpy as np

class Support_Vector_Machine:
    def __init__(self, learning_rate, iterations, lambda_param): #lambda_param is the regularization parameter
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param

    def fit(self, X, Y):
        self.m, self.n = X.shape # m is my number of entries and n is number of X features
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        #implementing gradient descent
        for i in range(self.iterations):
            self.update_weights()

    def update_weights(self):
        # label encodingy 
        y_label = np.where(self.Y <= 0, -1, 1)

        for index, x_i in enumerate(self.X):
            condition = y_label[index] * (np.dot(x_i,self.w) + self.b) >= 1
            if condition:
                dw = 2 * self.lambda_param * self.w
                db = 0
            else:
                dw = 2 * self.lambda_param * self.w -  y_label[index] * x_i
                db = y_label[index]

            #updating weights
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self, X):
        output = np.dot(X, self.w) + self.b

        predicted_labels = np.sign(output)

        y_cap = np.where(predicted_labels <= -1, 0, 1)

        return y_cap