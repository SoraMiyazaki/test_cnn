import numpy as np
import pandas as pd

pred_list = pd.read_csv("pred.csv", header=None).values

pred_list= pred_list[:,::2]

pred_arr = []
for row in range(len(pred_list)):
    pred_col = []
    for col in range(len(pred_list[1])):
        pred_num = float(pred_list[row,col].replace("tensor(", ""))
        pred_col.append(pred_num)
    pred_arr.append(pred_col)
    
pred_max = []    
for row in range(len(pred_arr)):
    pred = max(pred_arr[row])
    pred_max.append(pred)
    
pred = []  
for row in range(len(pred_arr)):
    count = 0
    for col in range(len(pred_arr[1])):
        count += 1
        if pred_arr[row][col] == pred_max[row]:
            pred.append(count)

truth = pd.read_csv("../data/poc/test.csv", header=None).values
truth = np.delete(truth, 0, 1)
pred = np.array([pred]).T

result = np.hstack([pred, truth])

np.savetxt("result.csv", result, fmt='%d')

print(type(result))
print(np.abs(pred - truth))  