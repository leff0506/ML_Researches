import pandas as pd
import numpy as np
import sklearn.cluster as sk_cluster
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import sklearn.preprocessing as sk_preprocessing
import scipy.cluster.hierarchy as sp_clustering_hr
import sklearn.decomposition as sk_decomposition
def get_pivot_table(model,X,clients_df):
    model.fit(X)

    clients_df["class"] = model.labels_
    client_pivot_table = clients_df.pivot_table(index="class", values=["average_price", "return_rate", "overall_rating",
                                                                       "customer_id"],
                                                aggfunc={"average_price": np.mean, "return_rate": np.mean,
                                                         "overall_rating": np.mean, "customer_id": len})
    return client_pivot_table


clients_df = pd.read_csv("../data/customer_online_closing_store.csv")
columns = clients_df.columns.values
clients_df["return_rate"] = clients_df["items_returned"]/clients_df["items_purchased"]
clients_df["average_price"] = clients_df["total_spent"]/clients_df["items_purchased"]

# print(clients_df[["average_price","return_rate","overall_rating"]])

X = np.array(clients_df[["average_price","return_rate","overall_rating"]]).reshape(-1,3)

min_max_scaler = sk_preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# pca_tranformer = sk_decomposition.PCA(n_components=2)
# X = pca_tranformer.fit_transform(X)
# print(X)

# linkage_method = "ward"
# dendrogram = sp_clustering_hr.dendrogram(sp_clustering_hr.linkage(X,method = linkage_method))
# agglomerative_model = sk_cluster.AgglomerativeClustering(n_clusters=4,linkage=linkage_method)

k_means_model = sk_cluster.KMeans(n_clusters=5,init = "k-means++",n_init= 10)

print(get_pivot_table(k_means_model,X,clients_df))

# plt.show()