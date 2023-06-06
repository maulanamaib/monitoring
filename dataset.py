import joblib
import pandas as pd

def modell(x):
    # import data test
    # cols = ["Age","Sex","ALT","AST"]
    cols = ["Suhu(termometer)","Suhu(sensor)","pH(ph meter)","pH(sensor)"]
    df = pd.DataFrame([x],columns=cols)
    data_test = pd.read_csv('data/data.csv')
#     data_test = data_test.drop(data_test.columns[0],axis=1)
    # data_test = data_test.drop(data_test.columns[7:13],axis=1)
    # memasukkan data kedalam data test
    data_test = data_test.append(df,ignore_index = True)
    # print(data_test.columns)
    # print(data_test)
    # return data_test yang sudah dinormalisasi
    return joblib.load('data/nana.sav').fit_transform(data_test)

def svr(x):
    
    return joblib.load('data/SVRmodel.pkl').predict(x)
