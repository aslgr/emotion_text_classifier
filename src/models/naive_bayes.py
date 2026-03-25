from sklearn.naive_bayes import MultinomialNB

def train_naive_bayes(X_train, y_train, **kwargs):
    model = MultinomialNB(**kwargs)
    model.fit(X_train, y_train)
    return model