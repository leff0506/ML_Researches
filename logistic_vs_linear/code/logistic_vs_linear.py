import pandas as pd
import sklearn.linear_model as sk_lm
import sklearn.model_selection as sk_ms
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt


def get_prediction_and_proba_regr(X, y):
    regressor = sk_lm.LinearRegression()
    probas = sk_ms.cross_val_predict(regressor, X, y, cv=4)
    predictions = []
    for p in probas:
        if p >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions, probas


def get_prediction_and_proba_log(X, y):
    regressor = sk_lm.LogisticRegression()
    probas = sk_ms.cross_val_predict(regressor, X, y, cv=4, method="predict_proba")
    probas_res = []
    for i in probas:
        probas_res.append(i[1])
    predictions = sk_ms.cross_val_predict(regressor, X, y, cv=4)
    return predictions, probas_res


def comparison(X, y):
    lin = get_prediction_and_proba_regr(X, y)
    log = get_prediction_and_proba_log(X, y)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    ax1.set_xlabel("grade")
    ax1.set_ylabel("linear proba")

    ax2.set_xlabel("grade")
    ax2.set_ylabel("linear predicted")

    ax3.set_xlabel("grade")
    ax3.set_ylabel("logistic proba")

    ax4.set_xlabel("grade")
    ax4.set_ylabel("logistic predicted")

    print("linear:")
    print(sk_metrics.confusion_matrix(y, lin[0]))
    print("Accuracy:", sk_metrics.accuracy_score(y, lin[0]))
    print("Recall:", sk_metrics.recall_score(y, lin[0]))
    print("Precision:", sk_metrics.precision_score(y, lin[0]))
    print("Logistic Loss:", sk_metrics.log_loss(y, lin[1]))
    ax1.scatter(X[:], lin[1])
    ax2.scatter(X[:], lin[0])
    print()

    print("Confusion matrics for logistic:")
    print(sk_metrics.confusion_matrix(y, log[0]))
    print("Accuracy:", sk_metrics.accuracy_score(y, log[0]))
    print("Recall:", sk_metrics.recall_score(y, log[0]))
    print("Precision:", sk_metrics.precision_score(y, log[0]))
    print("Logistic Loss:", sk_metrics.log_loss(y, log[1]))
    print("------------------------------------------\n")

    ax3.scatter(X[:], log[1])
    ax4.scatter(X[:], log[0])


df = pd.read_csv("../data/single_grade.csv")
columns = df.columns.tolist()
X = df[columns[:-1]]
y = df[columns[-1]]

print("single_grade_csv")
comparison(X, y)
# plt.show()

df = pd.read_csv("../data/linear_vs_logistic.csv")
columns = df.columns.tolist()
X = df[columns[:-1]]
y = df[columns[-1]]

print("linear_vs_logistic_csv")
comparison(X, y)
plt.show()