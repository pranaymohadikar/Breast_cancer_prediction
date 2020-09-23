from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)

model = pickle.load(open('gau.pickle', 'rb'))

@app.route('/')	
def home():

	return render_template('home.html')


@app.route('/predict',methods=['POST'])	
def predict():
	
	if request.method=='POST':
		mean_radius = (request.form['mean_radius'])
		mean_texture = (request.form['mean_texture'])
		mean_perimeter = (request.form['mean_perimeter'])	
		mean_area = (request.form['mean_area'])	
		mean_smoothness = (request.form['mean_smoothness'])

		

		prediction=	model.predict([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]])
		output=prediction[0]

	return render_template('index.html', prediction_cancer=f' answer is {output}')


if __name__=='__main__':
	app.run(debug=False)