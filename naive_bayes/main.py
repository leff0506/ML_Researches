from my_naive_bayes.bayes import CatBayes, MixedBayes
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection as ms
import sklearn.metrics as sk_metrics
import sklearn.naive_bayes as sk_naive_bayes
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# data = [[1,1,1],[1,0,1],[0,1,0],[0,2,0]]
# data_df = pd.DataFrame(data,columns=["a","b","C"])
# data_df.to_csv("data.csv",index=False)

data_df = pd.read_csv("../data/iris.csv")


columns = data_df.columns.tolist()
# for column in columns:
label_encoder = LabelEncoder()
data_df[columns[-1]] = label_encoder.fit_transform(data_df[columns[-1]])

# sns.pairplot(data_df,hue = columns[-1])
# plt.show()
X = data_df[columns[:-1]]
y = data_df[columns[-1]]
#

bayes = MixedBayes([MixedBayes.gaussian_distribution, MixedBayes.gaussian_distribution,MixedBayes.gaussian_distribution,MixedBayes.gaussian_distribution])
bayes.fit(X, y)
v_predictions = ms.cross_val_predict(bayes,X,y,cv = 4)

print(sk_metrics.accuracy_score(y,v_predictions))



