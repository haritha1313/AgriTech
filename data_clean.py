# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 20:41:26 2018

@author: pegasus
"""

import pandas as pd
import numpy as np
import csv

#load csv files
market_price = pd.read_csv('data/market.csv')
standard_price = pd.read_csv('data/standard.csv')
coc = pd.read_csv('data/cost-of-cultivation.csv')

#list of considered crops
products = ["groundnut", "paddy", "rice", "wheat", "barley", 
            "jowar", "bajra", "maize", "ragi", "gram", "tur", 
            "mustard", "soyabean", "sunflower", 
            "cotton", "jute", "sugarcane"]
market_edited = market_price[market_price['commodity'].str.lower().isin([x.lower() for x in products])]

#take mean of all varieties of one crop
grouped_market = market_edited.groupby(['state', 'commodity']).agg(['mean']).reset_index()
grouped_market.columns = ["".join(x) for x in grouped_market.columns.ravel()]

#convert price in quintal to price in kg
grouped_market['modal_pricemean'] = grouped_market['modal_pricemean'].apply(lambda x: float(x)/100)
standard_price['price'] = standard_price['price'].apply(lambda x: float(x)/100)

#convert to dictionary of crop to profit or cost
std_prices = dict(zip(standard_price.standard, standard_price.price))
coc_dict = dict(zip(coc.crop, coc.cost))

#state vs crop matrix
yield_mat = pd.read_csv('data/state-crop-yield.csv')

#state and crop lists
states = yield_mat['state']
crops = list(yield_mat.columns)[1:]

yield_np_mat = yield_mat.as_matrix()

prediction_table = list()
prediction_table.append(["state", "crop", "profit"])

for i in range(0,yield_np_mat.shape[0]-1):
    for j in range(1,yield_np_mat.shape[1]):
        if(yield_np_mat[i][j]!=0.0):
	    #if yielding in current state then calculate profit
            rs_per_kg = grouped_market[(grouped_market['state']==states[i]) & (grouped_market['commodity'] == crops[j-1])]['modal_pricemean']
            if(rs_per_kg.shape[0]==1):
		#use market price data
                diff = ((yield_np_mat[i][j]) * rs_per_kg.values[0]) - coc_dict[str(crops[j-1].lower())]
                val = diff if diff>0.0 else 1.0
            else:
		#use approximated standard price
                diff = ((yield_np_mat[i][j]) * std_prices[str(crops[j-1].lower())]) - coc_dict[str(crops[j-1].lower())]
                val = diff if diff>0.0 else 1.0
            
	    #append all results to a table
            prediction_table.append([states[i], crops[j-1], val])
        else:
	    #states which cant produce the given crop
            print("**", states[i],crops[j-1])
            prediction_table.append([states[i], crops[j-1], -1])
            
#print(prediction_table)

#save to csv
with open("data/state-profit-data.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(prediction_table)
    
