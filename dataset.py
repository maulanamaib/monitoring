import joblib
import pandas as pd

def modell(x):
    # import data test
    cols = ["Suhu(termometer)","Suhu(sensor)","pH(ph meter)","pH(sensor)"]
    df = pd.DataFrame([x],columns=cols)
    data_test = pd.read_csv('data/data.csv')
    data_test = data_test.drop(data_test.columns[0],axis=1)
    # memasukkan data kedalam data test
    data_test = data_test.append(other = df,ignore_index = True)
    # # return data_test yang sudah dinormalisasi
    # convert data_test to float
    data_test = data_test.astype(float)    
    # print(data_test)
    return joblib.load('data/nana.sav').fit_transform(data_test)



def svr(x):

    return joblib.load('data/SVRmodel.pkl').predict(x)
