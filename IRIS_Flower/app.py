# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import joblib


df = pd.read_csv("car_price_cleaned.csv")

random_forest = joblib.load('car_price_prediction.sav')

def predict(data):
    return random_forest.predict(data) 
    
def main():
       
    with st.sidebar:
        selected = option_menu ('Multiple Car Prediction System',
                            
                            ['Home',
                             'Car Price Prediction',
                            'Dataset label encoded',
                            'Authors Names',
                            ],
                            icons = ['house','car-front','person','book'],
                            default_index=0
    
                            )
    
    if(selected == "Home") :
        st.title("Prediction du Prix de voitures en utilisant Le Machine Learning")
        
        st.subheader("Author : Group 1 ")
        st.write(" We are happy to see you reading our work. This is project given by our teacher. We are trying to build a Machine Learning Model  for Car price prediction. \n The program is deproloyed by using streamlit. Below are the links of our dataset and github page. \n It will be our pleasure to receive  your recommandations and advices to perfom the model and improve the streamlit app.")
        st.subheader("Dataset link  : https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge\n")
        st.subheader("Github link  : https://github.com/FoubaDev/Car_Price_Prediction.git \n")
       
    if(selected == "Dataset label encoded") :
        st.write(df)
        st.write(df.shape)
    if(selected == "Car Price Prediction") :
    
       

        col1, col2,col3,col4 = st.columns(4)
    
   
    
        with col1:
            Levy = st.number_input('Levy',key=1,min_value = 0)
    
        with col2:
           manufacturer = st.number_input('Manufacturer', min_value=0)   
            
            # Encode the user input
          
         #   manufacturer = st.text_input('Manufacturer')
    
        with col3:
            
            model_input = st.number_input('Model',min_value=0)

    
        with col4:
        
            prod_year = st.number_input('Production year',min_value=1900,max_value=2023,step=1)
    
        with col1:
        
            category = st.number_input('Category',min_value=0),
            
    
        with col2 :
            
            leather_interior = st.number_input('Leather_interior',min_value=0)
    
        with col3 :
            
          
            fuel_type = st.number_input('Fuel',min_value=0)
        
           
    
        with col4:
        
            engine_volume = st.number_input('Engine Volume',key=2,min_value = 0)
    
        with col1 :
        
            mileage = st.number_input('Mileage',key=3,min_value = 0)
    
        with col2 :
        
            cylinder = st.number_input('Cylinder',key=4,min_value = 0)
    
        with col3 :
            
            gear_box_type = st.number_input('Gear box type',min_value=0)
        
        with col4:
            
            drive_wheels = st.number_input('Drive Wheels',min_value=0)
    
        with col1 :
          
            doors = st.number_input('Value of Doors',min_value=0)
            
        with col2 :
                
                wheel = st.number_input('Value of wheel',min_value=0)
                
    
        with col3 :
           
            color = st.number_input('Value of color',min_value=0)
            
    
        with col4 :
        
            airbags = st.number_input('Airbags',key=5,min_value = 0)
   
        my_price = ''
            # button of testx
        
        if st.button('Prédictions'):
            
            input_features =  pd.DataFrame([[Levy,manufacturer,model_input,prod_year,category,leather_interior,fuel_type,engine_volume,mileage,cylinder,gear_box_type,drive_wheels,doors,wheel,color,airbags]])
            result = predict(input_features)
            st.write("Cette voiture peut coûter :",result)
        st.success(my_price)
        
    if(selected == "Authors Names") :
         st.write("Julie  ALLARABAYE \n")
         st.write()
         st.write("LAGRE GABBA Bertrand \n")
         st.write("MBAIYAM DJENGOMDE Hipolite \n")
         st.write("MAHAMAT SALEH Yacoub \n")
     
          
if __name__=='__main__':
    main()
