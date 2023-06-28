# import requests
# from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
import os
path='output_all_students_Train_v10.xlsx'



def prepare_data(path):
    data = pd.read_excel(path)
    data["price"] = data["price"].astype(str)  
    data["price"] = data["price"].apply(lambda x: re.sub(r'\D', '', x) if x else '') 
    data["price"] = pd.to_numeric(data["price"])
    data.dropna(subset = ['price'], inplace = True)
    data['Area '] = pd.to_numeric(data['Area'], errors='coerce')
    data['city_area'] = data['city_area'].apply(lambda x: re.sub(r'\d+', '', x) if isinstance(x, str) else x)
    data['Street'] = data['Street'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    data['description '] = data['description '].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    data["Area"]=data["Area"].apply(lambda x: re.findall(r"\b\d+\b",str(x))[0] if len(re.findall(r"\b\d+\b",str(x)))==1 else None)
    data['Area'] = data['Area'].astype(float)
    data['floor'] = data['floor_out_of'].str.extract(r'קומה\s(\d+)')
    data['total_floors'] = data['floor_out_of'].str.extract(r'מתוך\s(\d+)')
    data["City"]=data['City'].replace("נהרייה","נהריה")
    data["City"]=data['City'].replace(" נהריה","נהריה")
    from datetime import datetime
    data['entranceDate '] = data['entranceDate '].replace('גמיש', 'flexible')
    data['entranceDate '] = data['entranceDate '].replace('גמיש ', 'flexible')
    

    data['entranceDate '] = data['entranceDate '].replace('לא צויין', 'not_defined')
    data['entranceDate '] = data['entranceDate '].replace('מיידי', 'Less_than_6 months')

# Convert valid date strings to datetime objects, others will be set as NaT
    valid_dates_mask = pd.to_datetime(data['entranceDate '], errors='coerce').notna()

# Get current date
    current_date = pd.to_datetime(datetime.now().date())

# Calculate time difference in months between entrance_date and current_date for valid date values
    data.loc[valid_dates_mask, 'time_difference'] = (current_date - pd.to_datetime(data.loc[valid_dates_mask, 'entranceDate '])).dt.days / 30

# Categorize the time difference into bins for valid date values
    bins = [-float('inf'), 6, 12, float('inf')]
    labels = ['Less_than_6 months', 'months_6_12', 'Above_year']
    data.loc[valid_dates_mask, 'entranceDate '] = pd.cut(data.loc[valid_dates_mask, 'time_difference'], bins=bins, labels=labels)

# Replace remaining values with appropriate labels
    data['entranceDate '] = data['entranceDate '].fillna('invalid_value')

# Drop the unnecessary column
    data = data.drop(['time_difference'], axis=1)
    boolean_columns = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']
    data[boolean_columns] = data[boolean_columns].astype(str)
    replacement_dict = {'יש': 1, 'יש ממ״ד': 1, 'יש מרפסת': 1, 'יש מיזוג אוויר': 1,'יש מיזוג אויר': 1, 'נגיש לנכים': 1,
                    'נגיש': 1,"לא נגיש":0 ,'yes': 1, 'TRUE': 1, 'True': 1, 'יש מחסן': 1, 'יש סורגים': 1,
                    'יש חנייה': 1,'יש חניה': 1, 'יש מעלית': 1, 'אין': 0, 'לא': 0, 'אין חניה': 0,
                    'אין ממ״ד': 0, 'אין מרפסת': 0, 'אין מחסן': 0, 'אין סורגים': 0,
                    'אין מעלית': 0, 'אין מיזוג אויר': 0, 'לא נגיש לנכים': 0, 'no': 0,'לא':0,
                    'FALSE': 0, 'False': 0,'כן':1, 'יש ממ״ד':1, 'יש ממ"ד':1, 'אין ממ"ד':0,"nan":0
                    }
    data[boolean_columns] = data[boolean_columns].replace(replacement_dict)
    data["room_number"]=data["room_number"].apply(lambda x: str(x))
    data["room_number"]=data["room_number"].apply(lambda x: re.sub(r"[^0-9.]", "", x))
    data["room_number"]=data["room_number"].apply(lambda x: float(x)  if x != '' else None)
    data["floor"]=data["floor"].astype(float)


    new_data=data[["City","Area","type","floor","room_number","furniture ","condition ","entranceDate ","hasElevator ", 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ',"price"]]
    new_data['Area'] = new_data['Area'].fillna(new_data['Area'].mean())
    new_data["floor"] = new_data["floor"].fillna(new_data['floor'].mean())


    #check for correlation between features
    corr_matrix = new_data[["Area","room_number","floor"]].corr(method='spearman')
    # Create a heatmap using the correlation matrix
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True)
     # we can see there is a high correlation between room number and area so we will remoove one of them

     
    encoded_df=new_data[["Area","price","floor",'hasParking ','hasBalcony ','hasAirCondition ','hasElevator ']]
    #columns_to_encode = ["City","type","condition "]
    #encoded_df = pd.get_dummies(new_data[columns_to_encode], prefix=columns_to_encode, prefix_sep='_', dtype=int,dummy_na=True)
    encoded_df["Area"]=new_data["Area"].values
    encoded_df["price"]=new_data["price"].values
    encoded_df["floor"]=new_data["floor"].values
    encoded_df['hasParking ']=new_data['hasParking '].values
    encoded_df['hasBalcony ']=new_data['hasBalcony '].values
    encoded_df['hasAirCondition ']=new_data['hasAirCondition '].values
    encoded_df['hasElevator ']=new_data['hasElevator '].values
    
#We couldn't deal with the categorical values in our flask application, 
#so we removed them in order to allow the prediction to run properly.

    #columns_to_drop=["City_nan",'type_nan','condition _nan']
    #encoded_df=encoded_df.drop(columns=columns_to_drop) #remoove one column for each type in order to avoid multicollinearity
    x=encoded_df.drop("price",axis=1).values
    y=encoded_df["price"].values

    return encoded_df,x,y
    



path='output_all_students_Train_v10.xlsx'
encoded_df,x,y=prepare_data(path)
print(encoded_df)
