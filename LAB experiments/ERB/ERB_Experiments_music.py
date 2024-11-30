#%%
!pip install river
!pip install deepctr-torch
#%%
import numpy as np
np.float = float
from river.drift import ADWIN
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
import keras
import json

import sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
#%%
#loading the data
reviews = pd.read_json("/Users/jasperbruin/Documents/driftwatch/LAB experiments/datasets/CDs_and_Vinyl.jsonl", lines=True)
#%%
#preparing the data
reviews = reviews[reviews["rating"] != 3]
reviews = reviews[reviews["timestamp"] > "2015-01-01"]
reviews["year"] = reviews["timestamp"].apply(lambda x: x.year)
reviews["month"] = reviews["timestamp"].apply(lambda x: x.month)
reviews["day"] = reviews["timestamp"].apply(lambda x: x.day)
#%%
#create reference dataframe
reference_data = reviews[reviews["timestamp"] > "2015-01-01"]
reference_data = reference_data[reference_data["timestamp"] < "2019-01-01"]
#create evaluation dataframe
evaluation_data = reviews[reviews["timestamp"] >= "2019-01-01"]
evaluation_data = evaluation_data.set_index(evaluation_data["timestamp"])
evaluation_data.sort_index(inplace = True)
#%%
# remove products that appear after ref
evaluation_data_with_old_products = evaluation_data[evaluation_data["parent_asin"].isin(reference_data["parent_asin"])]
#%%
#encode features in the reviews for use in ml model
sparse_features = ["parent_asin", "user_id","year","month","day"]
for feat in sparse_features:
        lbe = LabelEncoder()
        reviews[feat] = lbe.fit_transform(reviews[feat])
#%%
def create_matrix(df):
	
	N = len(df['user_id'].unique())
	M = len(df['product_id'].unique())
	
	# Map Ids to indices
	user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
	product_mapper = dict(zip(np.unique(df["product_id"]), list(range(M))))
	
	# Map indices to IDs
	user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
	product_inv_mapper = dict(zip(list(range(M)), np.unique(df["product_id"])))
	
	user_index = [user_mapper[i] for i in df['user_id']]
	product_index = [product_mapper[i] for i in df['product_id']]

	X = csr_matrix((df["rating"], (product_index, user_index)), shape=(M, N))
	
	return X, user_mapper, product_mapper, user_inv_mapper, product_inv_mapper
#%%
#rename for use in function
reference_data["product_id"] = reference_data["parent_asin"]
#%%
X, user_mapper, product_mapper, user_inv_mapper, product_inv_mapper = create_matrix(reference_data)
#%%
#train ml model
kNN = NearestNeighbors(n_neighbors=5, algorithm="brute", metric='cosine')
kNN.fit(X)
#%%
def find_similar_products(product_id, X, k, metric='cosine', show_distance=False):

  neighbour_ids = []

  product_ind = product_mapper[product_id]
  product_vec = X[product_ind]
  k+=1
  product_vec = product_vec.reshape(1,-1)
  neighbour = kNN.kneighbors(product_vec, return_distance=show_distance)
  for i in range(0,k):
    n = neighbour.item(i)
    neighbour_ids.append(product_inv_mapper[n])
  neighbour_ids.pop(0)
  return neighbour_ids
#%%
#sample reviews from products
sampled = evaluation_data_with_old_products.sample(100)

# sampled = evaluation_data_with_old_products.sample(10000)
#%%
#save sample
sampled.reset_index(inplace=True,drop=True)
sampled.to_json("sampled_products.json")
#%%

#%%
#store the predictions of the recommendation system (takes a long time)
preds = []
for i in range(len(sampled)):
    preds.append(find_similar_products(sampled["parent_asin"].values[i],X,4))
#%%
sampled.sort_index(inplace = True)
#%%
#store the recommendations in dataframe
sampled["recommendation"] = preds
#%%
reviews_copy = pd.read_json("/Users/jasperbruin/Documents/driftwatch/LAB experiments/datasets/CDs_and_Vinyl.jsonl/CDs_and_Vinyl.jsonl", lines=True)
#%%
#ensure same strutcture as ctr-predictor training data
reviews_copy = reviews_copy[reviews_copy["rating"] != 3]
reviews_copy = reviews_copy[reviews_copy["timestamp"] > "2015-01-01"]
#%%
# create a mapper for the ctr-predictor
music_mapper = dict(zip(reviews_copy["parent_asin"].unique(), reviews["parent_asin"].unique()))
#%%
# create a mapper for the ctr-predictor
music_user_mapper = dict(zip(reviews_copy["user_id"].unique(), reviews["user_id"].unique()))
#%%
# load the ctr-predictor
model = torch.load("./model_formatted_music_v2")
#%%
# calculate click-through-rate for the recommendations
def mean_predicted_ctr():
    res = []
    for i in range(len(sampled)):
        temp = []
        for j in range(len(sampled["recommendation"][i])):
            temp.append(model.predict({"parent_asin": pd.Series(music_mapper[sampled["recommendation"][i][j]]),"user_id": pd.Series(music_user_mapper[sampled["user_id"][i]]), "year": pd.Series(sampled["year"][i] - 2015), "month": pd.Series(sampled["month"][i] - 1), "day": pd.Series(sampled["day"][i] - 1)}))
        res.append(temp)    
    return res
#%%

#%%
ctr_list = mean_predicted_ctr()
#%%
ctr_list
#%%
means = []
for i in range(len(ctr_list)):
    means.append(np.mean(ctr_list[i]))
#%%

#%%
np.mean(means)
#%%
series_means = pd.Series(means, index=sampled["timestamp"])
#%%
rolling_means = series_means.rolling("30D").mean()
#%%
rolling_means.plot()
#%%
# create labels for the drift-detectors
correct_recommendations = []
for i in range(len(ctr_list)):
    for j in range(4):
        correct_recommendations.append(random.choices([0,1],weights=[1-ctr_list[i][j][0][0],ctr_list[i][j][0][0]])[0])
#%%

from river.drift import DDM
from river.drift import EDDM
#%%
#testing the different erb-detectors
#%%
adwin = ADWIN()
entry_number = 0
entries = []
for i in range(len(correct_recommendations)):
    adwin.add_element(correct_recommendations[i])
    if adwin.detected_change():
        print('Change detected in data: ' + str(correct_recommendations[i]) + ' - at index: ' + str(i) + ' - at entry: '+ str(entry_number))
        entries.append(entry_number)
    if i % 4 == 3:
        entry_number = entry_number + 1
#%%
for i in entries:
    print(sampled["timestamp"][i])
#%%
np.datetime64("2019-06-01") +  296 * np.timedelta64(1, 'D')
#%%
np.datetime64('2020-07-07') - np.datetime64('2020-03-23')
#%%
ddm = DDM(min_num_instances=30)
alerts_ddm = []
entry_number = 0
entries = []
for i in range(len(correct_recommendations)):
    ddm.add_element(1 - correct_recommendations[i])
    #if ddm.detected_warning_zone():
        #print('Warning zone has been detected in data: ' + str(correct_recommendations[i]) + ' - of index: ' + str(i) + ' - at entry: '+ str(entry_number))
    if ddm.detected_change():
        print('Change has been detected in data: ' + str(correct_recommendations[i]) + ' - of index: ' + str(i) + ' - at entry: '+ str(entry_number))
        entries.append(entry_number)
    if i % 4 == 3:
        entry_number = entry_number + 1
#%%
for i in entries:
    print(sampled["timestamp"][i])
#%%
np.datetime64('2020-07-07') - np.datetime64('2020-03-23')
#%%
ddm = EDDM()
alerts_ddm = []
entry_number = 0
entries = []
for i in range(len(correct_recommendations)):
    ddm.add_element(1 - correct_recommendations[i])
    #if ddm.detected_warning_zone():
        #print('Warning zone has been detected in data: ' + str(correct_recommendations[i]) + ' - of index: ' + str(i) + ' - at entry: '+ str(entry_number))
    if ddm.detected_change():
        print('Change has been detected in data: ' + str(correct_recommendations[i]) + ' - of index: ' + str(i) + ' - at entry: '+ str(entry_number))
        entries.append(entry_number)
    if i % 4 == 3:
        entry_number = entry_number + 1
#%%
counter = 0
for i in entries:
    print(sampled["timestamp"][i])
    if np.datetime64(sampled["timestamp"][i]) >= np.datetime64('2020-03-23'):
        counter = counter + 1
#%%
counter
#%%
