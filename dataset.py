import joblib
import pandas as pd

def modell(x):
    # import data test
    cols = ["Suhu(termometer)","Suhu(sensor)","pH(ph meter)","pH(sensor)"]
    data_test = pd.read_csv('data/data.csv')

    # membuat mean Suhu(termometer) dan pH(ph meter)
    suhuter = data_test['Suhu(termometer)'].mean()
    phmet = data_test['pH(ph meter)'].mean()
    # masukkan mean kedalam data x
    x.insert(0,suhuter)
    x.insert(2,phmet)
    
    df = pd.DataFrame([x],columns=cols)
    print(df)
    data_test = data_test.drop(data_test.columns[0],axis=1)
    # memasukkan data kedalam data test
    data_test = data_test.append(other = df,ignore_index = True)
    # # return data_test yang sudah dinormalisasi
    # convert data_test to float
    data_test = data_test.astype(float)    
    # print(data_test)
    return joblib.load('data/nana.sav').fit_transform(data_test)



def svr(x):
    print(x)
    return joblib.load('data/SVRmodel.pkl').predict(x)
