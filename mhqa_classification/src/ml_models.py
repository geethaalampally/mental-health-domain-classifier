from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


def train_ml_models(X_train, y_train):
    models = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['lr'] = lr

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models['nb'] = nb

    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, y_train)
    models['rf'] = rf

    return models