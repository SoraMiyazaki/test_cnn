import numpy as np
import pandas as pd
import glob
import re
import time
import cv2
import tqdm
import statistics
import csv

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# dataset
data_hd = sorted(glob.glob("C:/Users/mizutani/Desktop/CNN_pra/height/height_h_d/*.csv"), key=natural_keys)
data_hs = sorted(glob.glob("C:/Users/mizutani/Desktop/CNN_pra/height/height_h_s/*.csv"), key=natural_keys)
data_ud = sorted(glob.glob("C:/Users/mizutani/Desktop/CNN_pra/height/height_u_d/*.csv"), key=natural_keys)
data_us = sorted(glob.glob("C:/Users/mizutani/Desktop/CNN_pra/height/height_u_s/*.csv"), key=natural_keys)

# test
# data_hd = pd.read_csv("C:/Users/mizutani/Desktop/CNN_pra/height/height_h_d/hhd10_1057.csv", header=None).values
# data_hs = pd.read_csv("C:/Users/mizutani/Desktop/CNN_pra/height/height_h_s/hhs10_1057.csv", header=None).values
# data_ud = pd.read_csv("C:/Users/mizutani/Desktop/CNN_pra/height/height_u_d/hud10_1057.csv", header=None).values
# data_us = pd.read_csv("C:/Users/mizutani/Desktop/CNN_pra/height/height_u_s/hus10_1057.csv", header=None).values

waterline_height_avg = []

def draw(data):
    a = pd.read_csv(data, header=None).values
    # a = data
    a1 = np.round(a, decimals=0)

    img = np.zeros((250, 512))

    for i in range(img.shape[1]): #img.shape[1]
        for j in range(img.shape[0]):
            if j == int(a1[i]):
                img[j-1, i] = 1   

    waterline_height = []
    
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if img[j, i] == 1:
                waterline_height.append(j+1)

    waterline_height_avg.append(int(statistics.mean(waterline_height[0:50])))
    waterline_height_avg.append(int(statistics.mean(waterline_height[50:150])))
    waterline_height_avg.append(int(statistics.mean(waterline_height[100:200])))
    waterline_height_avg.append(int(statistics.mean(waterline_height[150:250])))
    waterline_height_avg.append(int(statistics.mean(waterline_height[200:300])))
    waterline_height_avg.append(int(statistics.mean(waterline_height[250:350])))
    waterline_height_avg.append(int(statistics.mean(waterline_height[300:400])))
    waterline_height_avg.append(int(statistics.mean(waterline_height[350:450])))
    waterline_height_avg.append(int(statistics.mean(waterline_height[400:500])))
    
# dataset
for i in range(10569):
    draw(data_hd[i])
    print("1:", i+1)

for i in range(10569):
    draw(data_hs[i])   
    print("2:",i+1) 

for i in range(10569):
    draw(data_ud[i])
    print("3:",i+1)

for i in range(10569):
    draw(data_us[i])
    print("4:",i+1)

# test
# draw(data_hd)
# draw(data_hs)
# draw(data_ud)
# draw(data_us)

feature_list = []

# dataset
file_hd = sorted(glob.glob("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\image\\feature\\division_hd\\*.jpg"), key=natural_keys)
file_hs = sorted(glob.glob("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\image\\feature\\division_hs\\*.jpg"), key=natural_keys)
file_ud = sorted(glob.glob("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\image\\feature\\division_ud\\*.jpg"), key=natural_keys)
file_us = sorted(glob.glob("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\image\\feature\\division_us\\*.jpg"), key=natural_keys)

# test
# file_hd = sorted(glob.glob("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\image\\test\\hd\\feature\\*.jpg"), key=natural_keys)
# file_hs = sorted(glob.glob("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\image\\test\\hs\\feature\\*.jpg"), key=natural_keys)
# file_ud = sorted(glob.glob("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\image\\test\\ud\\feature\\*.jpg"), key=natural_keys)
# file_us = sorted(glob.glob("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\image\\test\\us\\feature\\*.jpg"), key=natural_keys)

for i in range(len(file_hd)):
    a = file_hd[i].replace("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\", "")
    print(a)
    feature_list.append(a)

for i in range(len(file_hs)):
    a = file_hs[i].replace("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\", "")
    print(a)
    feature_list.append(a)  

for i in range(len(file_ud)):
    a = file_ud[i].replace("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\", "")
    print(a)
    feature_list.append(a)

for i in range(len(file_us)):
    a = file_us[i].replace("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\", "")
    print(a)
    feature_list.append(a) 

# print(len(waterline_height_avg))
# print(len(feature_list))

np.savetxt("waterline_height_avg.csv", waterline_height_avg, fmt="%s")
np.savetxt("feature.csv", feature_list, fmt="%s")

truth = np.array([waterline_height_avg]).T
feature = np.array([feature_list]).T

ds = np.append(feature, truth, axis=1)

np.savetxt("dataset.csv", ds, fmt="%s")


data = pd.read_csv("dataset.csv", header=None).values.tolist()
df = []
for i in range(len(data)):
    a = "".join(data[i])
    b = a.split()
    df.append(b)

with open("C:/Users/mizutani/Desktop/code/data/poc/dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(df)
