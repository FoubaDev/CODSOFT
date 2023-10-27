# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu


@st.cache_resource
def load_data_csv():
   df = pd.read_csv("titanic.csv")

@st.cache_resource
def load_data_sav():
   loaded_model = joblib.load(open('titanic.sav', 'rb'))

df = load_data_csv()
loaded_model = load_data_sav()

@st.cache_resource
def predict(input_data):
    
    my_array = np.array(input_data)
    input_reshaped = my_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_reshaped)
    
    
    
    if (prediction[0] == 1):
        st.success("Survived")
    else:
        st.error("No survived")
        st.success(prediction)

@st.cache_resource
def bar_char_avg(arg1,arg2):
    st.write("""Average of year, grouped by Genre""")
    avg_age = df.groupby(arg1)[arg2].mean().round()
    avg_age = avg_age.reset_index()
    sex = avg_age[arg1]
    age_avg = avg_age[arg2]

    fig = plt.figure(figsize = (5, 4))

    plt.bar(sex, age_avg, color = 'maroon')
    plt.xlabel(arg1)
    plt.ylabel(arg2)
    plt.title(f"Matplotlib Bar Chart Showing the Average \
    '{arg2}' of titanic survival in Each '{arg1}'")
    st.pyplot(fig)

# create column for dashbaord
@st.cache_resource(experimental_allow_widgets=True)
def create_column():
    st.write('### Columns of different sizes')
    col1, col2 = st.columns([2,3])

    survived_list = df['Survived'].unique().tolist()
    embarqued_list = df['Embarked'].unique().tolist()
    sexe_list = df['Sex'].unique().tolist()

     #Configure and filter the slider widget for interactivity
    #survived_info = (df['Survived'].between(*survived_list))
    
    survived = st.selectbox('Choose a Survived', survived_list, 0)

    #create a multiselect widget to display genre
    embarked = st.multiselect('Choose Genre:',embarqued_list, default = ['C',\
                                         'Q',  'S'])
    #create a selectbox option that holds all unique years
    sex = st.selectbox('Choose a Sex',
    sexe_list, 0)

    new_sexe_age = (df['Age'].isin(sexe_list)) \
    & (df['Sex'] == sex)
    
    new_embarqued_age = (df['Age'].isin(embarqued_list)) \
    & (df['Embarked'] == embarked)
    
    new_survivde_sex = (df['Sex'].isin(survived_list)) \
    & (df['Survivde'] == survived)
    
    with col1:

        st.write("""#### Lists of movies filtered by year and Genre """)
        dataframe_sex_age = df[new_sexe_age]\
        .groupby(['Age',  'Sex'])['Survivid'].sum()

        dataframe_sex_age = dataframe_sex_age.reset_index()
        st.dataframe(dataframe_sex_age, width = 400)
    
    with col2:
        st.write("## hello")


def dashboard():
    # Sidebar - Filters
    columns = df.columns.drop(['PassengerId','Name','Cabin','Ticket'])
    selected_column = st.selectbox('Choose the feature to visualize', columns)

    # Display filtered df

    # Visualizations
    st.subheader('Data Visualization')

  
# Determine column type and create appropriate visualization
    if df[selected_column].dtypes in ['float64', 'int64']:
        # Histogram
        fig = plt.figure(figsize = (8,2))
        st.subheader('Histogram')
        histogram_data = df[selected_column]
        plt.hist(histogram_data, bins='auto')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)


    else:
        # Bar chart
        camembert(df,selected_column)
        st.subheader('Bar Chart')
        bar_chart_data = df[selected_column].value_counts()
        st.bar_chart(bar_chart_data)  

        st.subheader('Relation between Feature and Interpretation')
        st.area_chart(df[selected_column])
        

    # Footer
    st.sidebar.markdown('---')
    st.sidebar.markdown('@FoubaDev')

@st.cache_resource
def camembert(df,arg):
    fig, ax = plt.subplots(figsize=(10, 3))
    var_count = df[arg].value_counts()
    #st.write(var_count)
    var_values = var_count.index
    ax.pie(var_count, labels=var_values,autopct='%1.1f%%')
    plt.title(arg)
   # st.set_option('deprecation.showPyplotGlobalUse', False)
    ax.axis('equal')
    st.pyplot(fig)

@st.cache_resource(experimental_allow_widgets=True)
def choose_feature():
    embarqued_list = df['Embarked'].unique().tolist()
    sexe_list = df['Sex'].unique().tolist()
    #create a multiselect widget to display genre
    new_genre_list = st.multiselect('Choose Genre:',embarqued_list, default = ['C',\
                                         'Q',  'S'])
    #create a selectbox option that holds all unique years
    sex = st.selectbox('Choose a Year',
    sexe_list, 0)

# Function to plot the relation between two features using a scatter plot
@st.cache_resource
def show_relation(x_name, y_name):


# Display selected features
    st.subheader('Selected Features')


    # Plot bar chart
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x=x_name, hue='Survived', ax=ax2)
    plt.legend(title='Survived', loc='upper right')
    plt.xlabel(x_name)
    plt.ylabel('Count')
    st.pyplot(fig2)

def main():

    with st.sidebar:
        selected = option_menu ('Titatic survival Prediction System',
                            
                            ['Home',
                             'Correlation',
                             'Prediction',
                            'Dataset',
                            'Author',
                            ],
                            icons = ['house','bi bi-file-medical-fill','person','book','person'],
                            default_index=0
    
                            )
    
    if(selected == "Home") :
        st.title("Titanic survival Prediction Using Machine Learning")
        dashboard()


      
    if(selected == "Correlation") :
        st.header('Select Features')
        columns = df.columns.drop(['PassengerId','Name','Cabin','Ticket'])
        selected_feature_x = st.selectbox('Select X Feature', columns)
        selected_feature_y = st.selectbox('Select Y Feature', columns)
        st.subheader(f"Relation between {selected_feature_x} and {selected_feature_y}")

        show_relation(selected_feature_x, selected_feature_y)

        #bar_char_avg("Age","Fare")
        #st.write(df.shape)
        
    if(selected == "Dataset") :
        st.write(df)
        st.write(df.shape)
    if(selected == "Prediction") :
    
       
        col1, col2,col3,col4 = st.columns(4)
    
    
        with col1:
            passengerid = st.number_input('Passenger_Id', min_value=0) 
            pclass = st.number_input('Pclass', min_value=0) 
           

        with col2:
            sexe = ['Male','Female']

            sex = st.selectbox('Select the sex', sexe)
            sex_encoded = 0
            if sex == 'Male':
                sex_encoded = 1
            else:
                sex_encoded = 0
             
            age = st.number_input('Age',min_value=0)
            
        with col3:
            sibp = st.number_input('SibSp',min_value=0)
            parch = st.number_input('Parch',min_value=0)
        with col4:
            fare = st.number_input('Fare',min_value=0.0,step=0.01)

            emb = ['Q','S','C']
            embarked = st.selectbox('Select embarked value', emb)  

            emb_encoded = 0
            if embarked == 'Q':
                emb_encoded == 1
            elif embarked == 'S':
                emb_encoded = 2
            else:
                emb_encoded = 0
            


        titanic = ''
        data = (passengerid,pclass,sex_encoded,age,sibp,parch,fare,emb_encoded)
        
        if st.button('Prédictions'):
           
            result = predict(data)
        st.success(titanic)
        
    if(selected == "Author") :
         
         
         st.subheader("Author : LAGRE GABBA BERTRAND")
         st.write(" Data Develper") 
         st.write("Github link  : https://github.com/FoubaDev/CODSOFT.git \n")

         st.write(" It is my pleasure to see you reading my work. The objective of the dataset is to  predict Titatic survival.")

         st.write("Data scientist Intern at CodSoft \n")
        
if __name__=='__main__':
    main()
