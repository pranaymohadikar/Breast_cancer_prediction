from flask import Flask,render_template,request
import numpy as np
import pandas as pd

app=Flask(__name__)



@app.route('/')	
def home():

	return render_template('home.html')


@app.route('/predict',methods=['POST','GET'])	
def predict():
	
	if request.method=='POST':
		mean_radius = float(request.form['mean_radius'])
		mean_texture = float(request.form['mean_texture'])
		mean_perimeter = float(request.form['mean_perimeter'])	
		mean_area = float(request.form['mean_area'])	
		mean_smoothness = float(request.form['mean_smoothness'])

		model=pd.read_pickle('rfc_model.pickle')

		prediction=	model.predict([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]])
		output=prediction[0]

	return render_template('result.html', prediction_cancer=f' answer is {output}')


if __name__=='__main__':
	app.run(debug=True)