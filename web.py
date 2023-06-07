
import streamlit as st
import dataset
import time
import webbrowser
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import streamlit.components.v1 as components


st.write("")
    st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Tabel Sensor Suhu</h1>", unsafe_allow_html=True)

data = pd.read_csv('https://raw.githubusercontent.com/maulanamaib/monitoring/master/data/data.csv')
    data.fillna(0,inplace=True)
    data
    
data1 = dataset.modell(data)
prediksi = dataset.svr(data1)  
prediksi
