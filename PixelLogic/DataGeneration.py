import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import os
import csv


PATH = '../../../[Data] PixelLogic/PixelLogic.csv'
if not os.path.exists(PATH):
    pd.DataFrame(columns = [f'pixel{i}' for i in range(100)]+
                 [f'colsum{i}' for i in range(10)]+
                 [f'rowsum{i}' for i in range(10)]
                ).to_csv(PATH, index=False)
    
    
def gen_data_point(threshold):
    """
    returns: flattened-array, columnwise sum, rowwise sum
    """
    a = np.random.randn(10,10)
    min_val = a.min()
    max_val = a.max()
    a -= min_val
    a /= (max_val-min_val)
    a = a < threshold
    a = a.astype('int8')
    return np.concatenate((a.reshape(100), a.sum(0), a.sum(1)))


for threshold, num_points in tqdm(zip([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [49, 36, 25, 16, 9, 4, 1])):
    for i in trange(num_points*1_00_000):
        with open(PATH, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(gen_data_point(threshold))