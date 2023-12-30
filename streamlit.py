import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("Groceries data.csv")

df["month"].replace([i for i in range( 1 , 13)] ,["January","Febuary","March","April","May","June","July","August","September","October","November","December"], inplace=True)
df["day_of_week"].replace([i for i in range( 0 , 7)] ,["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], inplace=True)

st.title("Groceries dataset for Market Basket Analysis(MBA) Menggunakan Apriori Oleh Fauzan Fadhillah Arisandi")

def get_data(month='', day_of_week=''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month)) &
        (data["day_of_week"].str.contains(day_of_week)) 
    ]
    return filtered if filtered.shape[0] else "No Result!"

def user_input_features():
    item = st.selectbox("itemDescription",['tropical fruit', 'whole milk', 'pip fruit', 'other vegetables',
       'rolls/buns', 'pot plants', 'citrus fruit', 'beef', 'frankfurter',
       'chicken', 'butter', 'fruit/vegetable juice',
       'packaged fruit/vegetables', 'chocolate', 'specialty bar',
       'butter milk', 'bottled water', 'yogurt', 'sausage', 'brown bread',
       'hamburger meat', 'root vegetables', 'pork', 'pastry',
       'canned beer', 'berries', 'coffee', 'misc. beverages', 'ham',
       'turkey', 'curd cheese', 'red/blush wine',
       'frozen potato products', 'flour', 'sugar', 'frozen meals',
       'herbs', 'soda', 'detergent', 'grapes', 'processed cheese', 'fish',
       'sparkling wine', 'newspapers', 'curd', 'pasta', 'popcorn',
       'finished products', 'beverages', 'bottled beer', 'dessert',
       'dog food', 'specialty chocolate', 'condensed milk', 'cleaner',
       'white wine', 'meat', 'ice cream', 'hard cheese', 'cream cheese ',
       'liquor', 'pickled vegetables', 'liquor (appetizer)', 'UHT-milk',
       'candy', 'onions', 'hair spray', 'photo/film', 'domestic eggs',
       'margarine', 'shopping bags', 'salt', 'oil', 'whipped/sour cream',
       'frozen vegetables', 'sliced cheese', 'dish cleaner',
       'baking powder', 'specialty cheese', 'salty snack',
       'Instant food products', 'pet care', 'white bread',
       'female sanitary products', 'cling film/bags', 'soap',
       'frozen chicken', 'house keeping products', 'spread cheese',
       'decalcifier', 'frozen dessert', 'vinegar', 'nuts/prunes',
       'potato products', 'frozen fish', 'hygiene articles',
       'artif. sweetener', 'light bulbs', 'canned vegetables',
       'chewing gum', 'canned fish', 'cookware', 'semi-finished bread',
       'cat food', 'bathroom cleaner', 'prosecco', 'liver loaf',
       'zwieback', 'canned fruit', 'frozen fruits', 'brandy',
       'baby cosmetics', 'spices', 'napkins', 'waffles', 'sauces', 'rum',
       'chocolate marshmallow', 'long life bakery product', 'bags',
       'sweet spreads', 'soups', 'mustard', 'specialty fat',
       'instant coffee', 'snack products', 'organic sausage',
       'soft cheese', 'mayonnaise', 'dental care', 'roll products ',
       'kitchen towels', 'flower soil/fertilizer', 'cereals',
       'meat spreads', 'dishes', 'male cosmetics', 'candles', 'whisky',
       'tidbits', 'cooking chocolate', 'seasonal products', 'liqueur',
       'abrasive cleaner', 'syrup', 'ketchup', 'cream', 'skin care',
       'rubbing alcohol', 'nut snack', 'cocoa drinks', 'softener',
       'organic products', 'cake bar', 'honey', 'jam', 'kitchen utensil',
       'flower (seeds)', 'rice', 'tea', 'salad dressing',
       'specialty vegetables', 'pudding powder', 'ready soups',
       'make up remover', 'toilet cleaner', 'preservation products'])
    day_of_week = st.selectbox("Day Of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    month = st.select_slider("Month",[ "January","Febuary","March","April","May","June","July","August","September","October","November","December"])
    
    return day_of_week,month,item

day_of_week,month,item = user_input_features()

data = get_data(month,day_of_week)

def convert_values(value):
    if value <= 0:
        return 0
    elif value >=1:
        return 1
    
if type(data) != type ("No Result"):
    baskets=df.groupby(['Member_number','itemDescription'])['itemDescription'].count().unstack()
    baskets=baskets.fillna(0).reset_index()
    baskets = baskets.iloc[:, 1:baskets.shape[1]].applymap(convert_values)

    df_new = pd.DataFrame(baskets)
    freq_items = apriori(df_new, min_support=0.05, use_colnames=True, max_len=3).sort_values(by='support')

    rules=association_rules(freq_items, metric="lift", min_threshold=1).sort_values('lift',ascending=False)
    rules=rules[['antecedents','consequents','support','confidence','lift']]
    rules.sort_values('confidence', ascending=False, inplace=True)

def  parse_list(x):
    x=list(x)
    if len(x) ==1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)
    
def return_item_df(item_antecendents):
    data=rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    return list(data.loc[data["antecedents"]== item_antecendents].iloc[0,:])

if type(data) != type("No Result!"):
    st.markdown("Hasil Rekomendasi :")
    st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{return_item_df(item)[1]}** secara bersamaan")
