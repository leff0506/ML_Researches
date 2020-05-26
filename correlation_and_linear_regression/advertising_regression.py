import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import sklearn.model_selection as ms


def train_linear_model(X, y):
    linear_model = lm.LinearRegression()
    linear_model.fit(X, y)
    return linear_model


def get_MSE(model, X, y_true):
    y_predicted = model.predict(X)
    MSE = sm.mean_squared_error(y_true, y_predicted)
    return MSE


advertising_df = pd.read_csv("../data/advertising.csv")
ad_data = advertising_df[["TV", "radio", "newspaper"]]
sales = advertising_df[["sales"]]

labels = ad_data.columns.values
# print(labels)
X_train, X_test, y_train, y_test = ms.train_test_split(ad_data, sales, shuffle=True)

linear_regression = train_linear_model(X_train, y_train)

# print(get_MSE(linear_regression, ad_data, sales))
print("Full MSE:", get_MSE(linear_regression, X_test, y_test))
print()
for z in range(0, 3):
    feature_name = labels[z]
    print("{} remains".format(feature_name))
    print("Pearson correletion coeficient between {} and sales is {}".format(feature_name,
                                                                             np.corrcoef(ad_data[feature_name],
                                                                                         sales["sales"])[0][1]))
    model_1_feature = train_linear_model(X_train[[feature_name]], y_train)
    print("Mse:", get_MSE(model_1_feature, X_test[[feature_name]], y_test))
    print()
