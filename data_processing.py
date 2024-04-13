# Method below is based on make_opportunity

"""
Preprocess McRoberts data into 30s and 10s windows
with a 15s and 5 sec overlap respectively
Sample Rate: 50 Hz
https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition

For a typical challenge, 1 sec sliding window and 50%
overlap is used. Either specific sets or runs are used
for training or sometimes both run count and subject count are specified.


Usage:
    python make_mcroberts.py

"""

from scipy import stats as s
import numpy as np
import glob
import os
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed

label2code = {
    'no activity':0,
    'sitting':1,
    'standing':2,
    'walking':3,
    'running':4,
    'cycling':5,
    'stair_down':6,
    'stair_up':7,
    'elevator_down':8,
    'elevator_up':9,
    'escalator_down':10,
    'escalator_up':11,
    'dragging':12,
    'kicking':13,
    'punching':14,
    'throwing':15,
    'bus':16,
    'car':17,
    'tram':18,
    'train':19
}
code2label = {code: label for label, code in label2code.items()}


def content2x_and_y(data_content, epoch_len=30, sample_rate=100, overlap=15):
    sample_count = int(np.floor(len(data_content) / (epoch_len * sample_rate)))

    sample_label_idx = 14
    sample_x_back_acc_idx = 0
    sample_y_back_acc_idx = 1
    sample_z_back_acc_idx = 2
    sample_x_back_gyr_idx = 3
    sample_y_back_gyr_idx = 4
    sample_z_back_gyr_idx = 5
    sample_x_arm_acc_idx = 7
    sample_y_arm_acc_idx = 8
    sample_z_arm_acc_idx = 9
    sample_x_arm_gyr_idx = 10
    sample_y_arm_gyr_idx = 11
    sample_z_arm_gyr_idx = 12
    sample_back_atm_idx = 6


    sample_limit = sample_count * epoch_len * sample_rate
    data_content = data_content[:sample_limit, :]

    label = data_content[:, sample_label_idx]
    x_back_acc = data_content[:, sample_x_back_acc_idx]
    y_back_acc = data_content[:, sample_y_back_acc_idx]
    z_back_acc = data_content[:, sample_z_back_acc_idx]
    x_back_gyr = data_content[:, sample_x_back_gyr_idx]
    y_back_gyr = data_content[:, sample_y_back_gyr_idx]
    z_back_gyr = data_content[:, sample_z_back_gyr_idx]
    x_arm_acc = data_content[:, sample_x_arm_acc_idx]
    y_arm_acc = data_content[:, sample_y_arm_acc_idx]
    z_arm_acc = data_content[:, sample_z_arm_acc_idx]
    x_arm_gyr = data_content[:, sample_x_arm_gyr_idx]
    y_arm_gyr = data_content[:, sample_y_arm_gyr_idx]
    z_arm_gyr = data_content[:, sample_z_arm_gyr_idx]
    back_atm = data_content[:, sample_back_atm_idx]


    # to make overlappting window
    offset = overlap * sample_rate
    shifted_label = data_content[offset:-offset, sample_label_idx]
    shifted_x_back_acc = data_content[offset:-offset:, sample_x_back_acc_idx]
    shifted_y_back_acc = data_content[offset:-offset:, sample_y_back_acc_idx]
    shifted_z_back_acc = data_content[offset:-offset:, sample_z_back_acc_idx]
    shifted_x_back_gyr = data_content[offset:-offset:, sample_x_back_gyr_idx]
    shifted_y_back_gyr = data_content[offset:-offset:, sample_y_back_gyr_idx]
    shifted_z_back_gyr = data_content[offset:-offset:, sample_z_back_gyr_idx]
    shifted_x_arm_acc = data_content[offset:-offset:, sample_x_arm_acc_idx]
    shifted_y_arm_acc = data_content[offset:-offset:, sample_y_arm_acc_idx]
    shifted_z_arm_acc = data_content[offset:-offset:, sample_z_arm_acc_idx]
    shifted_x_arm_gyr = data_content[offset:-offset:, sample_x_arm_gyr_idx]
    shifted_y_arm_gyr = data_content[offset:-offset:, sample_y_arm_gyr_idx]
    shifted_z_arm_gyr = data_content[offset:-offset:, sample_z_arm_gyr_idx]
    shifted_back_atm = data_content[offset:-offset:, sample_back_atm_idx]


    shifted_label = shifted_label.reshape(-1, epoch_len * sample_rate)
    shifted_x_back_acc = shifted_x_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_back_acc = shifted_y_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_back_acc = shifted_z_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_back_gyr = shifted_x_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_back_gyr = shifted_y_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_back_gyr = shifted_z_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_arm_acc = shifted_x_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_arm_acc = shifted_y_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_arm_acc = shifted_z_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_arm_gyr = shifted_x_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_arm_gyr = shifted_y_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_arm_gyr = shifted_z_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_back_atm = shifted_back_atm.reshape(-1, epoch_len * sample_rate, 1)
    

    shifted_X = np.concatenate([shifted_x_back_acc, shifted_y_back_acc, shifted_z_back_acc,
                                shifted_x_back_gyr, shifted_y_back_gyr, shifted_z_back_gyr,
                                shifted_x_arm_acc,  shifted_y_arm_acc,  shifted_z_arm_acc,
                                shifted_x_arm_gyr,  shifted_y_arm_gyr,  shifted_z_arm_gyr,
                                shifted_back_atm], axis=2)

    label = label.reshape(-1, epoch_len * sample_rate)
    x_back_acc = x_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    y_back_acc = y_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    z_back_acc = z_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    x_back_gyr = x_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_back_gyr = y_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_back_gyr = z_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_arm_acc = x_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    y_arm_acc = y_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    z_arm_acc = z_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    x_arm_gyr = x_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_arm_gyr = y_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_arm_gyr = z_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    back_atm = back_atm.reshape(-1, epoch_len * sample_rate, 1)


    X = np.concatenate([x_back_acc, y_back_acc, z_back_acc,
                        x_back_gyr, y_back_gyr, z_back_gyr,
                        x_arm_acc,  y_arm_acc,  z_arm_acc,
                        x_arm_gyr,  y_arm_gyr,  z_arm_gyr,
                        back_atm], axis=2)

    X = np.concatenate([X, shifted_X])
    label = np.concatenate([label, shifted_label])
    return X, label


def clean_up_label(X, labels):
    # 2. majority voting for label in each epoch
    final_labels = []
    for i in range(labels.shape[0]):
        row = labels[i, :]
        final_labels.append(s.mode(row)[0])
    final_labels = np.array(final_labels, dtype=int)
    return X, final_labels

def process_row(row):
    # Apply LOWESS smoothing
    smoothed_row = lowess(row, np.arange(len(row)), frac=0.5)[:, 1]
    return smoothed_row


def post_process_oppo(X, y, pid):
    zero_filter = np.array(y != 0)

    X = X[zero_filter]
    y = y[zero_filter]
    pid = pid[zero_filter]

    # set mean pressure of each window to 0
    means = X[:, :, -1].mean(axis=1)
    X[:, :, -1] -= means[:, None]

    # Parallel processing for LOWESS
    print("Smoothing pressure values...")
    X[:, :, -1] = np.array(Parallel(n_jobs=-1)(delayed(process_row)(X[i, :, -1]) for i in tqdm(range(X.shape[0]))))

    return X, y, pid

def butterworth_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data


def process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap):
    X = []
    y = []
    pid = []
    sample_rate = 50

    for file_path in tqdm(file_paths):
        subject_id = int(file_path.split("_")[-1][2:4])
        
        # Reassign experiment 4 data to other subjects
        subject_id_map = {11: 3, 12: 4, 15: 5}
        if subject_id in subject_id_map:
            subject_id = subject_id_map[subject_id]

        file_type = file_path.split("_")[-2]
        exp_number = int(file_path.split("_")[-3][3:4])

        # "FreeLiving" data currently not analysed
        if file_type == "freeliving":
            continue

        # read in data
        datContent = pd.read_csv(file_path, 
                                 sep=",",
                                 parse_dates=["time"],
                                 index_col="time").iloc[:,1:]

        datContent["label activity"] = datContent["label activity"].map(label2code)
        datContent = datContent.to_numpy()

        current_X, current_y = content2x_and_y(
            datContent,
            sample_rate=sample_rate,
            epoch_len=epoch_len,
            overlap=overlap,
        )

        current_X, current_y = clean_up_label(current_X, current_y)
        
        if len(current_y) == 0:
            continue
        ids = np.full(
            shape=len(current_y), fill_value=subject_id, dtype=np.int32
        )
        if len(X) == 0:
            X = current_X
            y = current_y
            pid = ids
        else:
            X = np.concatenate([X, current_X])
            y = np.concatenate([y, current_y])
            pid = np.concatenate([pid, ids])

    # post-process
    y = y.flatten()
    X = X / 1  # convert to g
    # clip_value = 3
    # X = np.clip(X, -clip_value, clip_value)
    X, y, pid = post_process_oppo(X, y, pid)

    np.save(X_path, X)
    np.save(y_path, y)
    np.save(pid_path, pid)

    # print some dataset stats
    print("X shape:", X.shape)
    print("Y distribution:", len(set(y)))
    print(pd.Series(y).map(code2label).value_counts())
    print("User distribution:", len(set(pid)))
    print(pd.Series(pid).value_counts())



def get_write_paths(data_root):
    X_path = os.path.join(data_root, "X.npy")
    y_path = os.path.join(data_root, "Y.npy")
    pid_path = os.path.join(data_root, "pid.npy")
    return X_path, y_path, pid_path


def main():
    data_root = "Dataset/"
    subject_paths = os.listdir(data_root)
    folder_paths = [data_root+subject_path+"/Sensor gegevens/verwerkte sensor gegevens NFI/" for subject_path in subject_paths]
    file_paths = [glob.glob(folder_path + "*.csv") for folder_path in folder_paths]
    file_paths = [item for sublist in file_paths for item in sublist]

    print("Processing for 30sec window..")
    data_root = "data/clean/w30/"
    X_path, y_path, pid_path = get_write_paths(data_root)
    epoch_len = 30
    overlap = 15
    process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)

    print("Processing for 10sec window..")
    data_root = "data/clean/w10/"
    X_path, y_path, pid_path = get_write_paths(data_root)
    epoch_len = 10
    overlap = 5
    process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)

if __name__ == "__main__":
    main()
