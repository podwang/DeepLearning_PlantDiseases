# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:33:00 2020

@author: qwert
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# virtual test generation codes
cols = ['a', 'b', 'c', 'd']
rows = range(40)
virtual_rst = pd.DataFrame(index = rows, columns = cols) 

for row in virtual_rst.itertuples():
    for col_idx in cols:
        #virtual_rst.at[row[0],col_idx] = (row[0] + ord(col_idx))/150
        virtual_rst.at[row[0],col_idx] = 0.9 + np.random.normal(loc = 0,
                                                                 scale = 0.01)
        
virtual_rst.to_excel('test.xlsx')

df_ovft_study_rst = pd.read_excel('test.xlsx', index_col = 0)

markers = ['o','v','^','p','s','x']

plt.figure(dpi = 150)
    
for idx, col_idx in enumerate(df_ovft_study_rst.columns):
    plt.plot(df_ovft_study_rst.index, 
                 df_ovft_study_rst[col_idx],
                 '-'+ markers[idx],
                 linewidth = 0.8,
                 markersize = 6,
                 label = col_idx)

# for marker shapes, see https://matplotlib.org/3.2.1/api/markers_api.html#module-matplotlib.markers

plt.title('Overfitting Ablation Study')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim([0.85,1.00])
plt.show()

