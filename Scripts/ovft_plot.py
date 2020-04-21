# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:33:00 2020

@author: qwert
"""

import pandas as pd
import matplotlib.pyplot as plt

# virtual test generation codes
cols = ['a', 'b', 'c', 'd']
rows = range(40)
virtual_rst = pd.DataFrame(index = rows, columns = cols) 

for row in virtual_rst.itertuples():
    for col_idx in cols:
        virtual_rst.at[row[0],col_idx] = row[0] + ord(col_idx)
        
virtual_rst.to_excel('test.xlsx')

df_ovft_study_rst = pd.read_excel('test.xlsx', index_col = 0)

markers = [',','v','^','p','s','x']

for idx, col_idx in enumerate(df_ovft_study_rst.columns):
    plt.scatter(x = df_ovft_study_rst.index, 
                y = df_ovft_study_rst[col_idx],
                alpha = 0.8, marker = markers[idx],
                s = 16)
# for marker shapes, see https://matplotlib.org/3.2.1/api/markers_api.html#module-matplotlib.markers

plt.title('Overfit Ablation Study')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / %')
plt.legend()
plt.show()

