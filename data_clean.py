# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 20:41:26 2018

@author: pegasus
"""

import pandas as pd
import numpy as np
import csv

market_price = pd.read_csv('market.csv')
standard_price = pd.read_csv('standard.csv')

products = ["groundnut", "paddy", "rice", "wheat", "barley", 
            "jowar", "bajra", "maize", "ragi", "gram", "tur", 
            "mustard", "soyabean", "sunflower", 
            "cotton", "jute", "sugarcane"]
market_edited = market_price[market_price['commodity'].str.lower().isin([x.lower() for x in products])]

grouped_market = market_edited.groupby(['state', 'commodity']).agg(['mean']).reset_index()
grouped_market.columns = ["".join(x) for x in grouped_market.columns.ravel()]

#convert price in quintal to price in kg
grouped_market['modal_pricemean'] = grouped_market['modal_pricemean'].apply(lambda x: float(x)/100)
standard_price['price'] = standard_price['price'].apply(lambda x: float(x)/100)

std_prices = dict(zip(standard_price.standard, standard_price.price))

yield_mat = pd.read_csv('state-crop-yield.csv')
states = yield_mat['state']
crops = list(yield_mat.columns)[1:]

yield_np_mat = yield_mat.as_matrix()

prediction_table = list()

for i in range(1,yield_np_mat.shape[0]-1):
    for j in range(1,yield_np_mat.shape[1]):
        if(yield_np_mat[i][j]!=0.0):
            rs_per_kg = grouped_market[(grouped_market['state']==states[i]) & (grouped_market['commodity'] == crops[j-1])]['modal_pricemean']
            if(rs_per_kg.shape[0]==1):
                val = ((yield_np_mat[i][j]) * rs_per_kg).to_string
            else:
                val = ((yield_np_mat[i][j]) * std_prices[str(crops[j-1].lower())]).to_string
            
            prediction_table.append([states[i], crops[j-1], val])
        else:
            print("**", states[i],crops[j-1])
            
print(prediction_table)

with open("state-yield-data.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(prediction_table)