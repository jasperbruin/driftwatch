#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

from evidently.calculations.stattests.jensenshannon import jensenshannon_stat_test
from evidently.calculations.stattests.kl_div import kl_div_stat_test
from evidently.calculations.stattests import hellinger_stat_test
from evidently.metric_preset import DataDriftPreset
from evidently.options.data_drift import DataDriftOptions
from evidently.core import ColumnType
#%%
# read data from file
pd.options.mode.copy_on_write = True 
data = pd.read_json("Amazon_Fashion.jsonl", lines=True)

#create reference dataframe
reference_data = data[data["timestamp"] > "2015-01-01"]
reference_data = reference_data[reference_data["timestamp"] < "2020-01-01"]
reference_array = reference_data["rating"].values
reference_data = reference_data.set_index(reference_data["timestamp"])
reference_data.sort_index(inplace = True)

#create evaluation dataframe
evaluation_data = data[data["timestamp"] >= "2020-01-01"]
evaluation_data = evaluation_data.set_index(evaluation_data["timestamp"])
evaluation_data.sort_index(inplace = True)

#calculate threshold
rolling = reference_data["rating"].rolling("30D").mean()
significance_threshold = rolling.mean() - 1.96 * rolling.std()
#%%
# gets 30-day sliding_windows of a sorted by timestamp Dataframe
def get_sliding_window_with_newvals(x):
    windows = []
    times = []
    new_values = []
    drift_flag = []
    start_time = x["timestamp"].values[0]
    day = np.timedelta64(1, 'D')
    month = np.timedelta64(30, 'D')
    end_time = start_time + month
#iterates over the dataframe to extract 30 day windows, always advancing a day
    while end_time <= x["timestamp"].values[-1]:
        end_time = start_time + month
        window = x[x["timestamp"] >= start_time]
        new_values.append(window[window["timestamp"] < (start_time + day)]["rating"].values)
        window = window[window["timestamp"] < end_time]
        windows.append(window["rating"].values)
        times.append(end_time)
        drift_flag.append(np.mean(window["rating"]) < significance_threshold)
        start_time = start_time + day
        
    return times, windows , new_values, drift_flag
#%%
# gets the drift scores for an array of distributions
def get_drift(x, method):
    drift_scores = []
    for i in range(len(x)):
        
        if x[i].size == 0:
            drift_scores.append(0)
        else:
            drift_scores.append(method(reference_array,x[i]))
    return drift_scores
#%%
# gets the drift scores for an array of distributions (for evidently drift detector functions)
def get_drift_evidently(x, method):
    drift_scores = []
    for i in range(len(x)):
        if x[i].size == 0:
            drift_scores.append(0)
        else:
            drift_scores.append(method(pd.Series(reference_array),pd.Series(x[i]),ColumnType.Numerical,0.1))
    return drift_scores
#%%
# gets an array of correct and false predictions, real fpr from np.mean of result 
def get_fpr(drift_scores,threshold):
    correct_pred = []
    for i in range(first_drift):
        if drift_scores[i] >= threshold and not sliding_windows[3][i]:
            correct_pred.append(1)
        elif not sliding_windows[3][i]:
            correct_pred.append(0)
    return correct_pred
#%%
#calculate all the infos and sliding windows
sliding_windows = get_sliding_window_with_newvals(evaluation_data)
#%%
# first_drift contains batchnumber of drift in focus
flag = False
sec_flag = False
for i in range(len(sliding_windows[3])):
    if sliding_windows[3][i]:
        flag = True
    if flag:
        if not sliding_windows[3][i]:
            sec_flag = True
    if sec_flag and sliding_windows[3][i]:
        first_drift = i
        break
    
#%%
# calculating all False-Positive Rates of the detectors
#%%
window_drift_wasserstein = get_drift(sliding_windows[1],scipy.stats.wasserstein_distance)
fpr_wasserstein = get_fpr(window_drift_wasserstein,0.1)
print("FPR: " + str(np.mean(fpr_wasserstein)))
#%%
window_drift_ks = get_drift(sliding_windows[1],scipy.stats.ks_2samp)
for i in range(len(window_drift_ks)):
    window_drift_ks[i] = window_drift_ks[i].statistic
fpr_ks = get_fpr(window_drift_ks,0.05)
print("FPR: " + str(np.mean(fpr_ks)))
#%%
window_drift_energy = get_drift(sliding_windows[1],scipy.stats.energy_distance)
fpr_energy = get_fpr(window_drift_energy,0.1)
print("FPR: " + str(np.mean(fpr_energy)))
#%%
window_drift_jensen = get_drift_evidently(sliding_windows[1],jensenshannon_stat_test)
for i in range(len(window_drift_jensen)):
    window_drift_jensen[i] =window_drift_jensen[i].drift_score
fpr_jensen = get_fpr(window_drift_jensen,0.05)
print("FPR: " + str(np.mean(fpr_jensen)))
#%%
window_drift_kl = get_drift_evidently(sliding_windows[1],kl_div_stat_test)
for i in range(len(window_drift_kl)):
    window_drift_kl[i] =window_drift_kl[i].drift_score
fpr_kl = get_fpr(window_drift_kl,0.01)
print("FPR: " + str(np.mean(fpr_kl)))
#%%
window_drift_hell = get_drift_evidently(sliding_windows[1],hellinger_stat_test)
for i in range(len(window_drift_hell)):
    window_drift_hell[i] =window_drift_hell[i].drift_score
fpr_hell = get_fpr(window_drift_hell,0.05)
print("FPR: " + str(np.mean(fpr_hell)))
#%%
# calculates the detection latency with drift scores and threshold
def get_latency(drift, threshold):
    for i in range(first_drift,len(sliding_windows[1])):
        if drift[i] >= threshold:
            return i - first_drift
    return -1
#%%
get_latency(window_drift_wasserstein,0.1)
#%%
get_latency(window_drift_ks,0.05)
#%%
get_latency(window_drift_energy,0.1)
#%%
get_latency(window_drift_jensen,0.05)
#%%
get_latency(window_drift_kl,0.01)
#%%
get_latency(window_drift_hell,0.05)
#%%
