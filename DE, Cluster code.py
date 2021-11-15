# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:07:53 2020

@author: Dou
"""


##### Part1 Cluster 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

##########################
# Merge Dataset 
##########################

# read data and set missing value to 'nan' value
gdp_df = pd.read_csv(r"gdp_per_capita_yearly_growth.csv", usecols=["country", "2018"],
                         keep_default_na=False, na_values=[""])
income_df = pd.read_csv(r"income_per_person.csv", usecols=["country", "2018"],
                         keep_default_na=False, na_values=[""])
industry_percent_df = pd.read_csv(r"industry_percent_of_gdp.csv", usecols=["country", "2018"],
                         keep_default_na=False, na_values=[""])
inflation_df = pd.read_csv(r"inflation_annual_percent.csv", usecols=["country", "2018"],
                         keep_default_na=False, na_values=[""])
life_exp_df = pd.read_csv(r"life_expectancy_years.csv", usecols=["country", "2018"],
                         keep_default_na=False, na_values=[""])

# rename new columns
gdp_df.rename(columns={'2018':'gdp_per_cap'}, inplace=True)
income_df.rename(columns={'2018':'income_per_cap'}, inplace=True)
industry_percent_df.rename(columns={'2018':'industry_percent'}, inplace=True)
inflation_df.rename(columns={'2018':'inflation'}, inplace=True)
life_exp_df.rename(columns={'2018':'life_exp'}, inplace=True)

# merge all colums into one dataset 
merged_inner1 = pd.merge(gdp_df, income_df, on='country')
merged_inner2 = pd.merge(merged_inner1, industry_percent_df, on='country')
merged_inner3 = pd.merge(merged_inner2, inflation_df, on='country')
merged_total= pd.merge(merged_inner3, life_exp_df, on='country')

# to see data summary
print(merged_total.shape)
# to see data description
merged_total.describe()
print(merged_total)
################################################
# Data Cleaning - Replace missing value to mean 
################################################

import numpy as np
# import imputer module from Scikit-learn and instantiate imputer object
from sklearn.preprocessing import Imputer
# replce nan value to mean
imputer = Imputer(missing_values=np.nan, strategy='mean')

# define columns to impute on
cols = ['gdp_per_cap',
         'income_per_cap',
         'industry_percent',
         'inflation',
         'life_exp']

# fit imputer and transform dataset, store in df_country
out_imp = imputer.fit_transform(merged_total[cols])
df_country = pd.DataFrame(data = out_imp, columns = cols)
df_country = pd.concat([df_country, merged_total[['country']]], axis = 1)

# set country name as index
df_country.set_index('country')
# to see pairplots between varibles
sns.pairplot(data=df_country)


############################
# scaling and normalization
############################

# min-max normalization is one of the most popular scaling processes
# load module and instantiate scaler object
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# normalize the data and store in out_scaled numpy array
out_scaled = scaler.fit_transform(df_country[cols])
# print the normalization result
print(out_scaled)

###################################
# Reduce Feature Dimension -- PCA
###################################
from sklearn.decomposition import PCA
 
# reduce to two dimension
pca = PCA(n_components=2)
# PCA dimension reduction with standardized data
X_pca = pca.fit_transform(out_scaled)

# generate dataframe after using PCA
X_pca_frame = pd.DataFrame(X_pca, columns=['pca_1', 'pca_2'])
X_pca_frame.head()


##############################
# Cluster -- HCA vs. K-Means
#############################

### 1. HCA

# import metrics module
from sklearn import metrics
# import module and instantiate HCA object
from sklearn.cluster import AgglomerativeClustering

# create list of k values to test and then use for loop
n_clusters = [2,3,4,5,6,7,8]
for num in n_clusters:
    HCA = AgglomerativeClustering(n_clusters=num, 
                               affinity='euclidean', linkage='ward',
                               memory='./model_storage/dendrogram', 
                               compute_full_tree=True)
    cluster_labels= HCA.fit_predict(X_pca)
    S = metrics.silhouette_score(X_pca, cluster_labels)
    print("n_clusters = {:d}, silhouette score {:1f}".format(num, S))

'''
Output: n_clusters = 2, silhouette score 0.324499
        n_clusters = 3, silhouette score 0.311090
        n_clusters = 4, silhouette score 0.330543
        n_clusters = 5, silhouette score 0.314995
        n_clusters = 6, silhouette score 0.346927
        n_clusters = 7, silhouette score 0.342908
        n_clusters = 8, silhouette score 0.362966
'''


### 2. K-Means

# import KMeans module
from sklearn.cluster import KMeans

# create list of k values to test and then use for loop
n_clusters = [2,3,4,5,6,7,8]
for k in n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_pca)
    cluster_labels = kmeans.predict(X_pca)
    S = metrics.silhouette_score(X_pca, cluster_labels)
    print("n_clusters = {:d}, silhouette score {:1f}".format(k, S))

'''
Output: n_clusters = 2, silhouette score 0.390532
        n_clusters = 3, silhouette score 0.407605
        n_clusters = 4, silhouette score 0.395579
        n_clusters = 5, silhouette score 0.417256
        n_clusters = 6, silhouette score 0.373466
        n_clusters = 7, silhouette score 0.369369
        n_clusters = 8, silhouette score 0.377828
'''

# Thus we choose 5 as the best number of clusters
# tol: stop when the distance between centers of two adjacent clusters is less than 0.0001
# max_iter: stop when iteration reach to 500
clus_kmeans = KMeans(n_clusters=5, tol=0.0001, max_iter=500)
# fit to input data
kmeans =clus_kmeans.fit(X_pca)

# get cluster assignments of input data and print first ten results
df_country['K-means Cluster Labels'] = kmeans.labels_
print(df_country[:10])

# visualize the group of countries set with the cluster labels displayed
X_pca_frame['K-means Cluster Labels'] = kmeans.labels_
sns.lmplot(x='pca_1', y='pca_2', 
           hue="K-means Cluster Labels", data=X_pca_frame, fit_reg=False)

# save the results in csv file
df_country.to_csv(r'kmeans_cluster.csv', index = False)


##### Part2 Regression

##############################
# Reading Data
#############################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

df = pd.read_csv("data-us-old2.csv")

df.index.name = "record"
df.head()


##############################
# Linear Regression
#############################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

colY = ['L-m','L-f','Unemployment']
colX = ['US Covid Cases','Deaths']


X = df[colX].values
y = df[colY].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#instantiate regression object and fit to training data
clf = LinearRegression()
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
y_pred


##############################
# Lasso Regression
#############################

# import modules
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

#instantiate classifier object and fit to training data
clf = Lasso(alpha=0.3)
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
r2 = r2_score(y_test, y_pred) 
print('r2 score is = ' + str(r2))


##############################
# Ridge Regression 
#############################

# import modules
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

#instantiate classifier object and fit to training data
clf = Ridge(alpha=0.3)
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
r2 = r2_score(y_test, y_pred) 
print('r2 score is = ' + str(r2))






















