#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from flask import Flask, jsonify, request
from flask import Flask, render_template,request
import numpy as np
import pandas as pd
from datetime import date
import datetime
import joblib
import holidays

app = Flask(__name__)

def split(final_data):
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data['Year'] = final_data['Date'].dt.year
    final_data['Month']= final_data['Date'].dt.month
    final_data['Week'] = final_data['Date'].dt.week
    final_data['Day']  = final_data['Date'].dt.day
    return final_data

def holiday_in_week(final_data):
    dates =[]
    for ptr in holidays.US(years = 2010).items():
        dates.append(ptr[0])
    for ptr in holidays.US(years = 2011).items():
        dates.append(ptr[0])
    for ptr in holidays.US(years = 2012).items():
        dates.append(ptr[0])
    for ptr in holidays.US(years = 2013).items():
        dates.append(ptr[0])
        
    holiday_count=[] 
    for index, row in final_data.iterrows():
        dat = final_data['Date'][index]
        dt=[]
        for i in range(0,5):
            dt.append(dat - datetime.timedelta(days = i))
        for i in range(1,3):
            dt.append(dat + datetime.timedelta(days = i))
        count = 0
        for date in dates:
            if date in dt:
                count +=1
        holiday_count.append(count)
    return np.array(holiday_count)

def holiday_label(final_data):
    final_data.loc[(final_data.IsHoliday==True) ,'IsHoliday']= 1
    final_data.loc[(final_data.IsHoliday==False) ,'IsHoliday']= 0
    return final_data

def type_label(final_data):
    final_data.loc[(final_data.Type=='A') ,'Type']= 1
    final_data.loc[(final_data.Type=='B') ,'Type']= 2
    final_data.loc[(final_data.Type=='C') ,'Type']= 3
    return final_data

@app.route('/')
def index():
    return render_template('walmart1.html')

@app.route('/predict',methods=['POST'])
def predict(): 
    stores_data=pd.read_csv("stores.csv")
    
    final_data = pd.DataFrame()
    final_data['Store'] = np.array([int(request.form['Store'])])
    final_data['Dept'] = np.array([int(request.form['dept'])])
    final_data['Date'] = np.array([request.form['Date']])
    final_data['IsHoliday'] = np.array([bool(request.form['IsHoliday'])]) 

    final_data=split(final_data)
    final_data['Holidays'] = holiday_in_week(final_data)
    final_data=holiday_label(final_data) 
    final_data=final_data.reset_index(drop=True)
    final_data = final_data.merge(stores_data, on ='Store' , how = 'inner')
    final_data=type_label(final_data)
    
    final_data=final_data[['Store','Dept','IsHoliday','Size','Week','Type','Year','Holidays','Day']]
    regressor = joblib.load(r'C:\Users\80298\Desktop\applied ai\case study 1\11.Deployement\model.pkl')
    pred = regressor.predict(final_data)
    return render_template('result.html', variable=pred)

if __name__ == '__main__':
    app.run(debug=False)
