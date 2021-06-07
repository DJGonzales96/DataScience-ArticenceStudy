# -*- coding: utf-8 -*-
"""
Author: Devin Gonzales
Data Preparation/Overview
"""

import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read in data
df = pd.read_csv(r'Data Science Evaluation.csv')
df

# Overview of the data
df.info()

# Check for missing values
df.isnull().sum()

"""
## Optional: Data Scaling for Machine Processing
"""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

########################
####  Data Cleaning ####
########################

# Handle numerical attributes
imputer = SimpleImputer(strategy="median")
# remove the text attribute because median can only be calculated on numerical attributes
df_num = df.select_dtypes(include=['int64','float64'])
# fit the `imputer` instance to the training data
imputer.fit(df_num)
# Transform the training set
X = imputer.transform(df_num)

# Handle text/categorical attributes
df_cat = df.select_dtypes(include=['object'])
# convert categorical values into one-hot vectors
cat_encoder = OneHotEncoder(sparse=False)
df_cat_1hot = cat_encoder.fit_transform(df_cat)

# Pipeline for attributes, imputer isn't necessary but just in case
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), # fill in missing values
        ('std_scaler', StandardScaler()),              # feature scaling
    ])

df_prepared = num_pipeline.fit_transform(df_num)
cols=list(df.columns)

# Complete the pipeline with combine categorical attributes
num_attribs = list(df_num)
cat_attribs = list(df_cat)

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(sparse=False), cat_attribs),
    ])

df_prepared = full_pipeline.fit_transform(df)

# TODO (Optional): Convert the 2D array into Pandas DataFrame
cols=list(df.columns) + cat_encoder.categories_[0].tolist()

# Prepared data
df_prepared

"""
## Question 1
"""

# Sort values by profit
profit = df.sort_values(by='Profit as % of Cost', ascending=False)
profit

items = df.groupby('Item Type')
items.mean()[['Units Sold', 'Total Profit', 'Profit as % of Cost']].sort_values(by="Profit as % of Cost", ascending=False)

g = sns.catplot(
    y='Item Type', x = 'Profit as % of Cost',
    data=df,
    kind='point'
)

units_by_region = df.groupby(['Region'])['Units Sold'].sum().reset_index().sort_values('Units Sold', ascending=False)
units_by_region

g = sns.catplot(
    y='Region', x='Units Sold',
    data=units_by_region,
    kind='bar',
    aspect = 2
).set(title="# of Units Sold per Region", xlabel = "# of Units Sold", ylabel = "Region")

products = df.groupby(['Region','Item Type'])['Units Sold'].sum().reset_index().sort_values('Units Sold', ascending=False)
products.head()

# Plot Product Sales by Region
g = sns.catplot(
    y='Item Type', x = 'Units Sold', col = 'Region',
    col_wrap = 4,
    data=products,
    kind='bar',
    aspect = 0.8
).set(ylabel = 'Product', xlabel = '# of Units Sold')

"""
## Question 2
"""

from datetime import datetime

# Computes the days difference between two dates
def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%m/%d/%y")
    d2 = datetime.strptime(d2, "%m/%d/%y")
    return abs((d2 - d1).days)

# Add fulfillment speed to our dataset
df['Fulfillment Speed'] = df.apply(lambda x: days_between(x['Order Date'], x['Ship Date']), axis=1)
df.head()

# Look at the avg fulfillment speed of each product per region
products = df.groupby(['Region','Item Type'])['Fulfillment Speed'].mean().reset_index().sort_values('Fulfillment Speed', ascending=False)
products

# Bar plot to examine statistical spread of fulfillment speeds
sns.catplot(
    data = products,
    x="Fulfillment Speed",
    y="Region",
    kind="box"
)
plt.title("Average fulfillment speed per region");

# Plot Fulfillment Speed by Region
g = sns.catplot(
    y='Region', x = 'Fulfillment Speed', col = 'Item Type',
    col_wrap = 4,
    data=products,
    kind='bar',
    aspect = 0.8
).set(ylabel = 'Region', xlabel = 'Days to Fulfill Order')

# We'll use a relplot to examine multiple-categorical data
sns.relplot(
    data=df,
    y="Region",
    x="Units Sold",
    size="Profit as % of Cost",
    sizes=(2,150),
    height=10,
    hue="Fulfillment Speed"
)
plt.title("Releation between Units Sold and \nProfit as % of Cost per Region");
