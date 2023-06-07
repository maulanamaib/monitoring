
# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MTCPb7RxaVoxZDT0ywaa8zNgMfF54-Ev

# KLASIFIKASI WATER QUALITY

## Pengambilan Data
"""
import pandas as pd
import numpy as np
import dataset      
import streamlit as st
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

data1 = pd.read_csv('https://raw.githubusercontent.com/LALA09-erha/Python-TrainingProject/master/psd/Water%20Quality%20Testing.csv')
data2 = pd.read_csv('https://raw.githubusercontent.com/LALA09-erha/Python-TrainingProject/master/psd/waterquality.csv', encoding='windows-1252')

# menyeleksi hanya 500 data
data2 = data2.head(500)
# data 1 adalah manual sedangkan data 2 adalah data dari alat
print(len(data1) , len(data2))
with st.sidebar:
    selected = option_menu("Queens Gurame",["Cek Kualitas","Dataset", "Prepocessing", "Pelabelan", "Klasifikasi"],
  icons=["Arrows","book","cast", "book", "envelope"],
    )

# """## Pengumpulan Data"""
col1, col2 = st.columns(2)
with col1:
      suhuter = st.number_input("Masukkan Suhu(termometer)",min_value=0,max_value=50)
with col2:
      suhusen = st.number_input("Masukkan Suhu(sensor)",min_value=0, max_value=50)
    

#     bp = st.selectbox("Golongan Darah",("A","B","AB","O"))
col3,col4 =st.columns(2)
with col3:    
      phmet = st.number_input("Masukkan pH(ph meter)",min_value=0, max_value=50)
with col4:
      phsen = st.number_input("masukkan pH(ph sensor)", min_value=0 ,max_value=50)
columns = st.columns((2,3))


submit = columns[1].button("Submit")
# if sumbit and suhuter != 0 and suhusen != 0 and phmet != 0 and phsen != 0:
if submit:
    data = dataset.modell([suhuter,suhusen,phmet,phsen])
    data = dataset.cluster(data)
    print(data)

# suhumanual = data1['Temperature (°C)']
# mean = data2["TEMP"].mean()
# data2["TEMP"] = data2["TEMP"].replace(np.nan, mean)
# suhuauto = data2['TEMP']
# phmanual = data1['pH']
# phauto = data2['pH']
# # proses penggabungan data
# data = pd.concat([suhumanual.rename("Suhu(termometer)"), suhuauto.rename('Suhu(sensor)') ,phmanual.rename('pH(ph meter)'),phauto.rename('pH(sensor)')] , axis=1)
# X = data

# if selected == "Dataset":
#     _, col2, _ = st.columns([1, 1, 1])
#     with col2:
#         st.write('''## Dataset''')
#         st.write('''\n''')
#     st.write(X)

# # """## Proses Preprosessing"""

# if selected == "Prepocessing":
#     _, col2, _ = st.columns([1, 3, 1])
#     with col2:
#         st.write('''## Data setelah Prepocessing''')
#         st.write('''\n''')
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(X)
#     features_names = X.columns.copy()
#     scaled_features = pd.DataFrame(scaled, columns=features_names)
#     st.write(scaled_features.head(5))

# # """## Proses Pengelompokan data

# # Pelabelan Menggunakana KMeans Clustering
# # """

# # menentukan 
# scores = []
# # Number of clusters
# kmeans = KMeans(n_clusters=2)
# # Fitting the input data
# kmeans = kmeans.fit(X)
# # Getting the cluster labels
# labels = kmeans.predict(X)
# y = labels
# if selected == "Pelabelan":
#     _, col2, _ = st.columns([1, 2, 1])
#     with col2:
#         st.header('Data yang dilabeli')
#         st.write('''\n''')

#     label = pd.DataFrame(y)
#     label.rename(columns = {0: "Label"}, inplace=True)
#     new_data = pd.concat([data, label], axis=1)
#     st.write(new_data)

# # """## Eksekusi Ke model SVM"""

# if selected == "Klasifikasi":
#     _, col2, _ = st.columns([1, 2, 1])
#     with col2:
#         st.write("## Klasifikasi SVM")
#     # proses Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     # Fit model.
#     model = SVR()
#     model.fit(X_train, y_train)

#     # Make a prediction and determine the error.
#     y_test_prediction = model.predict(X_test)
#     y_test_prediction = (np.round(y_test_prediction))
#     test_rmse_svm = np.sqrt(accuracy_score(y_test, y_test_prediction))

#     # Result
#     st.write(f'svm-regressor-rmse  : {test_rmse_svm}')

