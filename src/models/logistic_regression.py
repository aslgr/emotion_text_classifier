from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, **kwargs):
    model = LogisticRegression(max_iter=1000, **kwargs)
    model.fit(X_train, y_train)
    return model