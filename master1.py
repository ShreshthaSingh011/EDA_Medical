import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from PIL import Image
#%matplotlib inline


st.title("Exploratory Data Analysis")
st.text('Objective: to predict medical charges with the help of other factors such as:')
st.text('age, sex, BMI, no. of children, smoking habits and region')
st.write('''Technologies Highlighted: Machine Learning, Linear Regression(Single and Multiple Features)
         Scikit Learn, Pandas,Numpy, Data Visualization techniques with matplotlib, seaborn etc.
         Correlation, Heatmaps, Various charts Feature Extraction ''')

st.text('# The Dataset')

df = pd.read_csv('insurance.csv')
st.write(df)

st.write("Dataset Description")
st.write(df.describe())

st.write("EDA over the dataset")

st.write("Distribution of Age")

fig_1 = px.histogram(df,
                   x='age',
                   marginal='box',
                   nbins=47,
                   title='Distribution of Age')
fig_1.update_layout(bargap=0.1)
st.write(fig_1)

st.write("Distribution of BMI")
#BMI
fig_2 = px.histogram(df,
                   x='bmi',
                   marginal='box',
                   color_discrete_sequence=['red'],
                   title='Distribution of BMI (Body Mass Index)')
fig_2.update_layout(bargap=0.1)
st.write(fig_2)

st.write("Medical Charges")
fig_3 = px.histogram(df,
                   x='charges',
                   marginal='box',
                   color='smoker',
                   color_discrete_sequence=['green', 'grey'],
                   title='Annual Medical Charges')
fig_3.update_layout(bargap=0.1)
st.write(fig_3)

st.write('Smoker count')
st.write(df.smoker.value_counts())

st.write(px.histogram(df, x='smoker', color='sex', title='Smoker'))

st.write("Age vs Charges")

# Age and Charges
fig_4 = px.scatter(df,
                 x='age',
                 y='charges',
                 color='smoker',
                 opacity=0.8,
                 hover_data=['sex'],
                 title='Age vs. Charges')
fig_4.update_traces(marker_size=5)
st.write(fig_4)

st.write('BMI and Charges')

fig_5 = px.scatter(df,
                 x='bmi',
                 y='charges',
                 color='smoker',
                 opacity=0.8,
                 hover_data=['sex'],
                 title='BMI vs. Charges')
fig_5.update_traces(marker_size=5)
st.write(fig_5)

st.write('Violin plot')
st.write(px.violin(df, x="children", y="charges"))

st.write("Correlation between BMI and Charges")
st.write(df.charges.corr(df.bmi))

smoker_values = {'no': 0, 'yes': 1}   # For categorical columns with words and categories
smoker_numeric = df.smoker.map(smoker_values)
st.write("Correlation berween Smokers and charges")
st.write(df.charges.corr(smoker_numeric))
st.write('Female : 0, Male : 1, Smoker : 1, Non-Smoker : 0, Areas: Northeast: 1,Northwesr: 2,Southeast: 3,Southwest: 4')
st.write("Correlation Matrix")
fig_5, ax = plt.subplots()
sns.heatmap(df.corr(), cmap='Reds', annot=True, ax=ax)
st.write(fig_5)
st.write('Linear Regression')
image = Image.open('./Result_charges.png')
st.image(image, caption='Results')
fig_6 = px.scatter_3d(df, x='age', y='bmi', z='charges')
fig_6.update_traces(marker_size=3, marker_opacity=0.5)
st.write('BMI vs age vs charges')
st.write(fig_6)
st.write("Result: The most heavily impacting factors over the charges are Age, BMI and Smoker/Non-Smoker ")
