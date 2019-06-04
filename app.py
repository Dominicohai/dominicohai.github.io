from flask import Flask, render_template, request
#from wtforms import Form, TextAreaField, validators

import pickle
import pandas as pd
import os


# Preparing the Model
cur_dir = os.path.dirname(__file__)
mlp = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model.pkl'), 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
	
	return render_template('index2.html')

@app.route('/results', methods=['POST'])
def predict():
    try:
    
        Date = (request.form['Date'])
        Hr_End= int(request.form['Hr_End'])
        Dry_Bulb= int(request.form['Dry_Bulb'])
        
        database=pd.read_csv('Dataset2.csv')


        input_data = [{'Date': Date, 'Hr_End': Hr_End, 'Dry_Bulb': Dry_Bulb}]
        
        data = pd.DataFrame(input_data)
        
        data['Date']=pd.to_datetime(data['Date'],format='%d/%m/%Y')
        data['id']=data['Date'] + pd.to_timedelta(data['Hr_End'],unit='h')
        database['id']=(pd.to_datetime(database['Date']) + pd.to_timedelta(database['Hr_End'],unit='h'))
        x=database[database['id']==data['id'][0]].index[0]

        database=database[(x-170):x+24]
        
        data['day_of_week']=data['Date'].dt.dayofweek
        data['Weekend']=round(data['day_of_week']/9,0)
        data['one_day_mean']=(database['System_Load'].rolling(window=24).mean())[x]
        data['shift_day']=(database['System_Load'].shift(24))[x]
        data['shift_week']=(database['System_Load'].shift(168))[x]
        data['relative']=(data['Date']-pd.to_datetime('2018/05/01'))/pd.Timedelta('1 day')
        data.fillna(0,inplace=True)
         
        data=data.drop(['Date','id'],axis=1)
        
        resfinal  = mlp.predict(data)
    
    except:
       resfinal=' unavailable. Kindly check your inputs.'
	
    return render_template('results.html', res=resfinal)

if __name__ == '__main__':
	app.run(port=7800,debug=True)
