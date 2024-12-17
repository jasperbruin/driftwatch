#%%
import numpy as np
import pandas as pd
import torch
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
#%%
torch.cuda.is_available()
#%%
#loading and preprocessing data
data = pd.read_json("Amazon_Fashion.jsonl", lines=True)
#%%
data = data[data["rating"] != 3]
#%%
data = data[data["timestamp"] > '2015-01-01']
#%%
data["year"] = data["timestamp"].apply(lambda x: x.year)
data["month"] = data["timestamp"].apply(lambda x: x.month)
data["day"] = data["timestamp"].apply(lambda x: x.day)
#%%
len(data)
#%%
#binarization of reviews to train ctr-predictor
data["rating"] = data["rating"].transform(lambda x: 1 if x > 3 else 0)
#%%
data.to_json("ml_data.json")
#%%
data.set_index(data["timestamp"],inplace=True)
#%%
data.sort_index(inplace=True)
#%%
data["rating"].rolling("30D").mean().plot()
#%%
# code based on an example in the deepctr-torch documentation
# generates a ctr-prediction model


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


if __name__ == "__main__":
    sparse_features = ["parent_asin", "user_id","year","month","day"]
    target = ['rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    fixlen_feature_columns = []
    fixlen_feature_columns.append(SparseFeat("parent_asin", 2255468, embedding_dim=8))
    fixlen_feature_columns.append(SparseFeat("user_id", 2255468, embedding_dim=8))
    fixlen_feature_columns.append(SparseFeat("year", 10000, embedding_dim=8))
    fixlen_feature_columns.append(SparseFeat("month", 20, embedding_dim=8))
    fixlen_feature_columns.append(SparseFeat("day", 40, embedding_dim=8))
    
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    data_sample = data
    
    # 3.generate input data for model
    model_input = {name: data_sample[name] for name in sparse_features}

    # 4.Define Model,compile and train

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)

    model.compile("adam", metrics=["AUC"], loss='binary_crossentropy')
    history = model.fit(model_input,data_sample[target].values,batch_size=1024,epochs=6,verbose=2,validation_split=0.1)
#%%
#testing the model
#%%
test_data = {"user_id": pd.Series([1615322]), "parent_asin": pd.Series([17530]),"year": pd.Series([18]), "month": pd.Series([4]),"day": pd.Series([1])}
#%%
model.predict(model_input)
#%%
np.mean(model.predict(model_input))
#%%
model.predict(test_data)
#%%
#saving the model
torch.save(model,"./model_formatted_fashion_cpu_v2")
#%%
#loading a different model
new_model = torch.load("./model")
#%%
