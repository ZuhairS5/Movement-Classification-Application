# Zhiyuan Wang, Zuhair Shaikh, Katherine Eriksson
# Team 11
# ELEC390 Final Project
# Department of Electrical and Computer Engineering
# Queen's University

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import random

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import*
from tkinter import filedialog
import tkinter as tk
from pathlib import PurePosixPath
import subprocess
import os
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


time_unit = 5
window_size = 20000

# Step 2 Data Storing
# read data
Kath_walk_front = pd.read_csv('KE_walk_f.csv')
Kath_walk_back = pd.read_csv('KE_walk_b.csv')
Kath_walk_jacket = pd.read_csv('KE_walk_j.csv')
Kath_jump_front = pd.read_csv('KE_jump_f.csv')
Kath_jump_back = pd.read_csv('KE_jump_b.csv')
Kath_jump_jacket = pd.read_csv('KE_jump_j.csv')
Zuhair_walk_front = pd.read_csv('ZS_walk_f.csv')
Zuhair_walk_back = pd.read_csv('ZS_walk_back.csv')
Zuhair_walk_jacket = pd.read_csv('ZS_walk_j.csv')
Zuhair_jump_front = pd.read_csv('ZS_jump_f.csv')
Zuhair_jump_back = pd.read_csv('ZS_jump_b.csv')
Zuhair_jump_jacket = pd.read_csv('ZS_jump_j.csv')
Zhiyuan_walk_front = pd.read_csv('ZW_walk_f.csv')
Zhiyuan_walk_back = pd.read_csv('ZW_walk_b.csv')
Zhiyuan_walk_jacket = pd.read_csv('ZW_walk_j.csv')
Zhiyuan_jump_front = pd.read_csv('ZW_jump_f.csv')
Zhiyuan_jump_back = pd.read_csv('ZW_jump_b.csv')
Zhiyuan_jump_jacket = pd.read_csv('ZW_jump_j.csv')


# shuffle function
def shuffle(data, time_unit):
    groups = []
    for start_time in range(0, int(data.iloc[-1]['Time (s)']), time_unit):
        end_time = start_time + time_unit
        group = data[(data['Time (s)'] >= start_time) & (data['Time (s)'] < end_time)]
        if not group.empty:
            groups.append(group)
    random.shuffle(groups)
    shuffled_df = pd.concat(groups)
    return shuffled_df


# divide data into 5s windows and shuffle
Kath_chunk_walk_front = shuffle(Kath_walk_front, time_unit)
Kath_chunk_walk_back = shuffle(Kath_walk_back, time_unit)
Kath_chunk_walk_jacket = shuffle(Kath_walk_jacket, time_unit)
Kath_chunk_jump_front = shuffle(Kath_jump_front, time_unit)
Kath_chunk_jump_back = shuffle(Kath_jump_back, time_unit)
Kath_chunk_jump_jacket = shuffle(Zuhair_jump_jacket, time_unit)
Zuhair_chunk_walk_front = shuffle(Zuhair_walk_front, time_unit)
Zuhair_chunk_walk_back = shuffle(Zuhair_walk_back, time_unit)
Zuhair_chunk_walk_jacket = shuffle(Zuhair_walk_jacket, time_unit)
Zuhair_chunk_jump_front = shuffle(Zuhair_jump_front, time_unit)
Zuhair_chunk_jump_back = shuffle(Zuhair_jump_back, time_unit)
Zuhair_chunk_jump_jacket = shuffle(Zuhair_jump_jacket, time_unit)
Zhiyuan_chunk_walk_front = shuffle(Zhiyuan_walk_front, time_unit)
Zhiyuan_chunk_walk_back = shuffle(Zhiyuan_walk_back, time_unit)
Zhiyuan_chunk_walk_jacket = shuffle(Zhiyuan_walk_jacket, time_unit)
Zhiyuan_chunk_jump_front = shuffle(Zhiyuan_jump_front, time_unit)
Zhiyuan_chunk_jump_back = shuffle(Zhiyuan_jump_back, time_unit)
Zhiyuan_chunk_jump_jacket = shuffle(Zhiyuan_jump_jacket, time_unit)

# combine chunks
chunks = pd.concat([Kath_chunk_walk_front, Kath_chunk_walk_back, Kath_chunk_walk_jacket,
                    Kath_chunk_jump_front, Kath_chunk_jump_back, Kath_chunk_jump_jacket,
                    Zuhair_chunk_walk_front, Zuhair_chunk_walk_back, Zuhair_chunk_walk_jacket,
                    Zuhair_chunk_jump_front, Zuhair_chunk_jump_back, Zuhair_chunk_jump_jacket,
                    Zhiyuan_chunk_walk_front, Zhiyuan_chunk_walk_back, Zhiyuan_chunk_walk_jacket,
                    Zhiyuan_chunk_jump_front, Zhiyuan_chunk_jump_back, Zhiyuan_chunk_jump_jacket])

# 90% for training and 10% for testing
# x_train -> 90% of chunks to train
x_train, x_test = \
             train_test_split(chunks, test_size=0.1, shuffle=True, random_state=0)
# print(x_train)

# store data in HDF5 file
with h5py.File('./hdf5_data.h5', 'w') as hdf:
    G1 = hdf.create_group('/Katherine')
    G1.create_dataset('Walk front', data=Kath_walk_front)
    G1.create_dataset('Walk back', data=Kath_walk_back)
    G1.create_dataset('Walk jacket', data=Kath_walk_jacket)
    G1.create_dataset('Jump front', data=Kath_jump_front)
    G1.create_dataset('Jump back', data=Kath_jump_back)
    G1.create_dataset('Jump jacket', data=Kath_jump_jacket)

    G2 = hdf.create_group('/Zuhair')
    G2.create_dataset('Walk front', data=Zuhair_walk_front)
    G2.create_dataset('Walk back', data=Zuhair_walk_back)
    G2.create_dataset('Walk jacket', data=Zuhair_walk_jacket)
    G2.create_dataset('Jump front', data=Zuhair_jump_front)
    G2.create_dataset('Jump back', data=Zuhair_jump_back)
    G2.create_dataset('Jump jacket', data=Zuhair_jump_jacket)

    G3 = hdf.create_group('/Zhiyuan')
    G3.create_dataset('Walk front', data=Zhiyuan_walk_front)
    G3.create_dataset('Walk back', data=Zhiyuan_walk_back)
    G3.create_dataset('Walk jacket', data=Zhiyuan_walk_jacket)
    G3.create_dataset('Jump front', data=Zhiyuan_jump_front)
    G3.create_dataset('Jump back', data=Zhiyuan_jump_back)
    G3.create_dataset('Jump jacket', data=Zhiyuan_jump_jacket)

    G41 = hdf.create_group('dataset')
    G41.create_dataset('Train', data=x_train)
    G41.create_dataset('Test', data=x_test)

# Read data out from HDF5 file
with h5py.File('./hdf5_data.h5', 'r') as hdf:
    column_names = ['Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
                    'Absolute acceleration (m/s^2)']

    KE_wf = hdf['/Katherine/Walk front']
    KE_wf_data = KE_wf[:]
    KE_wf_df = pd.DataFrame(KE_wf_data)
    KE_wf_df.columns = column_names

    KE_wb = hdf['/Katherine/Walk back']
    KE_wb_data = KE_wb[:]
    KE_wb_df = pd.DataFrame(KE_wb_data)
    KE_wb_df.columns = column_names

    KE_wj = hdf['/Katherine/Walk jacket']
    KE_wj_data = KE_wj[:]
    KE_wj_df = pd.DataFrame(KE_wj_data)
    KE_wj_df.columns = column_names

    KE_jf = hdf['/Katherine/Jump front']
    KE_jf_data = KE_jf[:]
    KE_jf_df = pd.DataFrame(KE_jf_data)
    KE_jf_df.columns = column_names

    KE_jb = hdf['/Katherine/Jump back']
    KE_jb_data = KE_jb[:]
    KE_jb_df = pd.DataFrame(KE_jb_data)
    KE_jb_df.columns = column_names

    KE_jj = hdf['/Katherine/Jump jacket']
    KE_jj_data = KE_jj[:]
    KE_jj_df = pd.DataFrame(KE_jj_data)
    KE_jj_df.columns = column_names

    ZS_wf = hdf['/Zuhair/Walk front']
    ZS_wf_data = ZS_wf[:]
    ZS_wf_df = pd.DataFrame(ZS_wf_data)
    ZS_wf_df.columns = column_names

    ZS_wb = hdf['/Zuhair/Walk back']
    ZS_wb_data = ZS_wb[:]
    ZS_wb_df = pd.DataFrame(ZS_wb_data)
    ZS_wb_df.columns = column_names

    ZS_wj = hdf['/Zuhair/Walk jacket']
    ZS_wj_data = ZS_wj[:]
    ZS_wj_df = pd.DataFrame(ZS_wj_data)
    ZS_wj_df.columns = column_names

    ZS_jf = hdf['/Zuhair/Jump front']
    ZS_jf_data = ZS_jf[:]
    ZS_jf_df = pd.DataFrame(ZS_jf_data)
    ZS_jf_df.columns = column_names

    ZS_jb = hdf['/Zuhair/Jump back']
    ZS_jb_data = ZS_jb[:]
    ZS_jb_df = pd.DataFrame(ZS_jb_data)
    ZS_jb_df.columns = column_names

    ZS_jj = hdf['/Zuhair/Jump jacket']
    ZS_jj_data = ZS_jj[:]
    ZS_jj_df = pd.DataFrame(ZS_jj_data)
    ZS_jj_df.columns = column_names

    ZW_wf = hdf['/Zhiyuan/Walk front']
    ZW_wf_data = ZW_wf[:]
    ZW_wf_df = pd.DataFrame(ZW_wf_data)
    ZW_wf_df.columns = column_names

    ZW_wb = hdf['/Zhiyuan/Walk back']
    ZW_wb_data = ZW_wb[:]
    ZW_wb_df = pd.DataFrame(ZW_wb_data)
    ZW_wb_df.columns = column_names

    ZW_wj = hdf['/Zhiyuan/Walk jacket']
    ZW_wj_data = ZW_wj[:]
    ZW_wj_df = pd.DataFrame(ZW_wj_data)
    ZW_wj_df.columns = column_names

    ZW_jf = hdf['/Zhiyuan/Jump front']
    ZW_jf_data = ZW_jf[:]
    ZW_jf_df = pd.DataFrame(ZW_jf_data)
    ZW_jf_df.columns = column_names

    ZW_jb = hdf['/Zhiyuan/Jump back']
    ZW_jb_data = ZW_jb[:]
    ZW_jb_df = pd.DataFrame(ZW_jb_data)
    ZW_jb_df.columns = column_names

    ZW_jj = hdf['/Zhiyuan/Jump jacket']
    ZW_jj_data = ZW_jj[:]
    ZW_jj_df = pd.DataFrame(ZW_jj_data)
    ZW_jj_df.columns = column_names

    DS_train = hdf['/dataset/Train']
    DS_train_data = DS_train[:]
    DS_train_df = pd.DataFrame(DS_train_data)
    DS_train_df.columns = column_names

    DS_test = hdf['/dataset/Test']
    DS_test_data = DS_test[:]
    DS_test_df = pd.DataFrame(DS_test_data)
    DS_test_df.columns = column_names


# Step 3 - Visualization

walk_df = pd.concat([KE_wf_df, KE_wb_df, KE_wj_df,
                     ZS_wf_df, ZS_wb_df, ZS_wj_df,
                     ZW_wf_df, ZW_wb_df, ZW_wj_df])

jump_df = pd.concat([KE_jf_df, KE_jb_df, KE_jj_df,
                     ZS_jf_df, ZS_jb_df, ZS_jj_df,
                     ZW_jf_df, ZW_jb_df, ZW_jj_df])

walk_df_shuff = shuffle(walk_df, time_unit)
jump_df_shuff = shuffle(jump_df, time_unit)


# function to generate line plots
def line_plot(dataset):
    time = dataset.iloc[0:-1, 0]
    x_accel = dataset.iloc[0:-1, 1]
    y_accel = dataset.iloc[0:-1, 2]
    z_accel = dataset.iloc[0:-1, 3]
    plt.plot(time, x_accel, label='Acceleration x')
    plt.plot(time, y_accel, label='Acceleration y')
    plt.plot(time, z_accel, label='Acceleration z')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.title('Line Plot for Accelerations Over Time')
    plt.legend()
    plt.show()


# function to generate scatter plots
def scatter_plot(dataset):
    time = dataset.iloc[0:-1, 0]
    x_accel = dataset.iloc[0:-1, 1]
    y_accel = dataset.iloc[0:-1, 2]
    z_accel = dataset.iloc[0:-1, 3]
    plt.scatter(time, x_accel, label='Acceleration x')
    plt.scatter(time, y_accel, label='Acceleration y')
    plt.scatter(time, z_accel, label='Acceleration z')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title('Scatter Plots for Accelerations Over Time')
    plt.legend()
    plt.show()


line_plot(walk_df_shuff)
line_plot(jump_df_shuff)
scatter_plot(walk_df_shuff)
scatter_plot(jump_df_shuff)


# Step 4 - Pre-processing
# function that uses moving average filter to reduce noise

# check if there are nan or '-' in the dataset
# numDash = (x_train == '-').sum().sum()
# numNan = x_train.isna().sum().sum()
# print('This is the number of dash: ', numDash)
# print('This is the number of nan: ', numNan)

def apply_moving_average_filter(df, window_size):
    # Create a new DataFrame to store the filtered values
    filtered_df = df.copy()

    # Apply moving average filter to acceleration columns
    filtered_df['Acceleration x (m/s^2)'] = df['Acceleration x (m/s^2)'].rolling(window=window_size, center=True, min_periods=1).mean()
    filtered_df['Acceleration y (m/s^2)'] = df['Acceleration y (m/s^2)'].rolling(window=window_size, center=True, min_periods=1).mean()
    filtered_df['Acceleration z (m/s^2)'] = df['Acceleration z (m/s^2)'].rolling(window=window_size, center=True, min_periods=1).mean()
    filtered_df['Absolute acceleration (m/s^2)'] = df['Absolute acceleration (m/s^2)'].rolling(window=window_size, center=True, min_periods=1).mean()

    # Return dataframe with filtered values
    return filtered_df


filtered_train_set = apply_moving_average_filter(DS_train_df, window_size)
# print(filtered_train_set)


# # normalize data function
# def normalize_data(df):
#
#     sc = preprocessing.StandardScaler()
#     normalized_data = sc.fit_transform(df)
#     normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
#
#     return normalized_df
#
#
# filtered_normalize_train_set = normalize_data(filtered_train_set)
# # print(filtered_normalize_train_set)


# Step 5 - Feature Extraction

# feature extraction for train data
featuresx_train = pd.DataFrame(columns = ['mean', 'std', 'median', 'sum of 1 std', 'max', 'min', 'range', 'variance', 'kurtosis', 'skew'])
featuresy_train = pd.DataFrame(columns = ['mean', 'std', 'median', 'sum of 1 std', 'max', 'min', 'range', 'variance', 'kurtosis', 'skew'])
featuresz_train = pd.DataFrame(columns = ['mean', 'std', 'median', 'sum of 1 std', 'max', 'min', 'range', 'variance', 'kurtosis', 'skew'])
featurestot_train = pd.DataFrame(columns = ['mean', 'std', 'median', 'sum of 1 std', 'max', 'min', 'range', 'variance', 'kurtosis', 'skew'])

featuresx_train['mean'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).mean()
featuresx_train['std'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).std()
featuresx_train['median'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).median()
featuresx_train['sum of 1 std'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).sum()
featuresx_train['max'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).max()
featuresx_train['min'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).min()
featuresx_train['range'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).max() - filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).min()
featuresx_train['variance'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).var()
featuresx_train['kurtosis'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).kurt()
featuresx_train['skew'] = filtered_train_set.iloc[0:-1, 1].rolling(window=window_size).skew()

featuresy_train['mean'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).mean()
featuresy_train['std'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).std()
featuresy_train['median'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).median()
featuresy_train['sum of 1 std'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).sum()
featuresy_train['max'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).max()
featuresy_train['min'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).min()
featuresy_train['range'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).max() - filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).min()
featuresy_train['variance'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).var()
featuresy_train['kurtosis'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).kurt()
featuresy_train['skew'] = filtered_train_set.iloc[0:-1, 2].rolling(window=window_size).skew()

featuresz_train['mean'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).mean()
featuresz_train['std'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).std()
featuresz_train['median'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).median()
featuresz_train['sum of 1 std'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).sum()
featuresz_train['max'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).max()
featuresz_train['min'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).min()
featuresz_train['range'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).max() - filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).min()
featuresz_train['variance'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).var()
featuresz_train['kurtosis'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).kurt()
featuresz_train['skew'] = filtered_train_set.iloc[0:-1, 3].rolling(window=window_size).skew()

featurestot_train['mean'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).mean()
featurestot_train['std'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).std()
featurestot_train['median'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).median()
featurestot_train['sum of 1 std'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).sum()
featurestot_train['max'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).max()
featurestot_train['min'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).min()
featurestot_train['range'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).max() - filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).min()
featurestot_train['variance'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).var()
featurestot_train['kurtosis'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).kurt()
featurestot_train['skew'] = filtered_train_set.iloc[0:-1, 4].rolling(window=window_size).skew()

# drop any null values
featuresx_train = featuresx_train.dropna()
featuresy_train = featuresy_train.dropna()
featuresz_train = featuresz_train.dropna()
featurestot_train = featurestot_train.dropna()


# # print jumping features
# print('Training Data X-Accel Features')
# print(featuresx_train)
# print(75 * '=')
# print('Training Data Y-Accel Features')
# print(featuresy_train)
# print(75 * '=')
# print('Training Data Z-Accel Features')
# print(featuresz_train)
# print(75 * '=')
# print('Training Data Total Accel Features')
# print(featurestot_train)


# # Step 6: Creating a classifier
# # For train
train_absolute_median = filtered_train_set['Absolute acceleration (m/s^2)'].median()
filtered_train_set.loc[filtered_train_set['Absolute acceleration (m/s^2)'] < train_absolute_median, 'Absolute acceleration (m/s^2)'] = 0
filtered_train_set.loc[filtered_train_set['Absolute acceleration (m/s^2)'] >= train_absolute_median, 'Absolute acceleration (m/s^2)'] = 1

train_data = filtered_train_set.iloc[:, :4]
train_labels = filtered_train_set.iloc[:, -1]

train_data_train, train_data_test, train_labels_train, train_labels_test = \
        train_test_split(train_data, train_labels, test_size=0.1, shuffle=True, random_state=0, stratify=train_labels)
# defining the classifier and the normalized inputs
sc = StandardScaler()
l_reg = LogisticRegression(max_iter=10000)
# training
clf = make_pipeline(StandardScaler(), l_reg)

clf.fit(train_data_train, train_labels_train)
train_labels_pred = clf.predict(train_data_test)
train_labels_clf_prob = clf.predict_proba(train_data_test)
print('Pred for train labels is: ', train_labels_pred )
print('Prb for train labels is: ', train_labels_clf_prob)
# obtaining classification accuracy
train_accuracy = accuracy_score(train_labels_test, train_labels_pred)
print('accuracy for train is: ', train_accuracy)
# obtaining the classification recall
train_recall = recall_score(train_labels_test, train_labels_pred)
print('recall for train is: ', train_recall)

# plotting the confusion matrix
train_cm = confusion_matrix(train_labels_test, train_labels_pred)
train_com_display = ConfusionMatrixDisplay(train_cm).plot()
plt.show()

# plotting the ROC curve
fpr, tpr, _ = roc_curve(train_labels_test, train_labels_clf_prob[:, 1], pos_label=clf.classes_[1])
train_roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# calculating the AUC
train_auc = roc_auc_score(train_labels_test, train_labels_clf_prob[:, 1])
print('the AUC for train is: ', train_auc)


def classifier(input_file, window):
    # Data pre-processing
    raw_data = pd.read_csv(input_file)

    # cut the raw data and put them into a list
    groups = []     # A list that contains all the data segments
    for start_time in range(0, int(raw_data.iloc[-1]['Time (s)']), time_unit):
        end_time = start_time + time_unit
        group = raw_data[(raw_data['Time (s)'] >= start_time) & (raw_data['Time (s)'] < end_time)].copy()
        if not group.empty:
            groups.append(group)

    # Apply moving average filter, normalize and add labels
    processed_groups = []
    for group in groups:
        filtered_group = apply_moving_average_filter(group, window_size)
        # normalized_group = normalize_data(filtered_group)
        median = filtered_group['Absolute acceleration (m/s^2)'].median()
        label = 'jumping' if median >= train_absolute_median else 'walking'
        filtered_group['label'] = label
        processed_groups.append(filtered_group)

    output_data = pd.concat(processed_groups)

    # Generate a random unique identifier for the output file name
    unique_file_found = False

    while not unique_file_found:
        random_integer = random.randint(0, 9999)

        output_file_name = f'result_{random_integer}.csv'

        # Check if the file with the generated name already exists
        if not os.path.exists(output_file_name):
            unique_file_found = True

    output_data.to_csv(output_file_name, index=False)

    # the figure that will contain the plot
    fig = Figure(figsize=(12, 4),
                 dpi=100)

    # list of squares
    y = [i ** 2 for i in range(101)]

    # adding the subplot
    plot1 = fig.add_subplot(121)
    plot2 = fig.add_subplot(122)

    # plotting the raw data
    plot1.scatter(raw_data.iloc[0:-1, 0], raw_data.iloc[0:-1, 4])
    plot1.set_xlabel('Time')
    plot1.set_ylabel('Acceleration')
    plot1.set_title('Scatter Plots for Accelerations Over Time')

    # plotting filtered and normalized data;
    plot2.scatter(filtered_group['Time (s)'], filtered_group['Absolute acceleration (m/s^2)'])
    plot2.set_xlabel('Time')
    plot2.set_ylabel('Acceleration')
    plot2.set_title('Filtered and Normalized Scatter Plots for Accelerations Over Time')

    canvas = FigureCanvasTkAgg(fig, master=window )
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack(pady=5)

    # inform user of results
    Label2 = Label(text="This person is: " + label, font=('Times', 20, 'bold'), fg="yellow", bg="#856ff8")
    Label2.pack(pady=6)

    # open new output file and display the name
    if os.path.exists(output_file_name):
        subprocess.run(['open', output_file_name])
        Label3 = Label(text="The new output file is named: " + output_file_name,  fg="white", bg="#856ff8")
        Label3.pack(pady=4)

# test====================================
# classifier('KE_walk_j.csv')
# classifier('KE_jump_j.csv')
# classifier('ZW_jump_j.csv')
# test_com = pd.concat([Zhiyuan_jump_jacket, Zuhair_walk_jacket])
# chunks.to_csv('chunks.csv')
# classifier('chunks.csv')


def plot_absolute_vs_time(csv_file):
    data = pd.read_csv(csv_file)

    jumping_data = data[data['labels'] == 'jumping']
    walking_data = data[data['labels'] == 'walking']

    plt.figure()

    plt.plot(jumping_data['time'], jumping_data['absolute'], color='blue', label='Jumping')
    plt.plot(walking_data['time'], walking_data['absolute'], color='green', label='Walking')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Absolute')

    plt.show()

#Step 7
# initialize name for button
name = "                 "

# create function from which user can select a csv file
def select_file():
    file_path = filedialog.askopenfilename()

    # if the correct file type is selected display the name
    if file_path.endswith('.csv'):
        print('Selected file:', file_path)
        name = PurePosixPath(file_path).name

        label = Label(text="Chosen File: " + name, fg="white", bg="#856ff8")
        label.pack(pady=4)

        # display button of which can generate output
        button2 = Button(
            text="Generate Output",
            width=15,
            height=4,
            bg="blue",
            fg="black",
            command=lambda: classifier(file_path, window)
        )
        button2.pack(pady=4)
        window.update()

    # incorrect file type has been selected, prompt user to correct this
    else:
        label = Label(text="Please select a CSV file", fg="white", bg="#856ff8")
        label.pack(pady=4)
        window.update()

# create GUI for user and format it
window = Tk()
window.title('ELEC 390 FINAL PROJECT - TEAM 11')
window.geometry("500x300+10+20")
window['background']='#856ff8'

welcome = Label(text="Are they walking or Jumping?", font=('Times', 20, "bold"), fg="yellow", bg="#856ff8")
welcome.pack()

# prompt user to select a file
button = Button(
    text="Select CSV file",
    width=15,
    height=4,
    bg="blue",
    fg="black",
    command=select_file,
)

button.pack(pady=4)

window.mainloop()