# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 20:41:26 2018

@author: pegasus
"""

import pandas as pd
import numpy as np

market_price = pd.read_csv('market.csv')
products = ["Groundnut", "paddy", "rice", "Wheat", "Barley", "Jowar", "Bajra", "Maize", "Ragi", "Gram", "Tur", "Rapeseed", "mustard", "Soyabean", "Sunflower", "Cotton", "Jute", "Sugar", "sugarcane"]
market_edited = market_price[market_price['commodity'].str.lower().isin([x.lower() for x in products])]

grouped_market = market_edited.groupby(['state', 'commodity']).agg(['mean']).reset_index()


grouped_market.columns = ["".join(x) for x in grouped_market.columns.ravel()]

#convert price in quintal to price in kg
grouped_market['modal_pricemean'] = grouped_market['modal_pricemean'].apply(lambda x: float(x)/100)
print(grouped_market)

yield_mat = pd.read_csv('state-crop-yield.csv')
states = yield_mat['state']
crops = list(yield_mat.columns)[1:]

yield_np_mat = yield_mat.as_matrix()

for i in range(35):
    for j in range(19):
        if(yield_np_mat[i][j]!=0.0):
            rs_per_kg = grouped_market[(grouped_market['state']==states[i]) & (grouped_market['commodity'] == crops[j-1])]['modal_pricemean']
            if(rs_per_kg.shape[0]==1):
                yield_np_mat[i][j] *= rs_per_kg
            else:
               # yield_np_mat[i][j] *= std
