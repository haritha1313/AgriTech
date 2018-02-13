# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:21:56 2018

@author: pegasus
"""

import pandas as pd

profit_data = pd.read_csv("data/state-profit-data.csv")

#print(profit_data.dtypes)

profit_data['state']=profit_data['state'].astype('category')
profit_data['crop'] = profit_data['crop'].astype('category')

profit_data['state_code'] = profit_data['state'].cat.codes
profit_data['crop_code'] = profit_data['crop'].cat.codes

mid = pd.unique(profit_data[['state', 'state_code']].values.ravel('K'))
state_encoding = pd.DataFrame({"state":mid[:35], "code":mid[35:]})

mid1 = pd.unique(profit_data[['crop', 'crop_code']].values.ravel('K'))
crop_encoding = pd.DataFrame({"crop":mid1[:17], "code":mid1[17:]})

state_encoding.to_csv("data/statecode.csv", index=False)
crop_encoding.to_csv("data/cropcode.csv", index = False)

dataset = pd.DataFrame({"state": profit_data["state_code"], "crop": profit_data["crop_code"], "profit": profit_data["profit"].apply(lambda x: int(x))})
columns = ["state", "crop", "profit"]
dataset[columns].to_csv("data/encoded_dataset.csv", index = False, header = False)
