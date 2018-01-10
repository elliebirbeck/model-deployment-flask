import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc


app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':

		# get uploaded image file if it exists
		file = request.files['image']
		if not file: return render_template('index.html', label="No file")
		
		# read in file as raw pixels values
		# (ignore extra alpha channel and reshape as its a single image)
		img = misc.imread(file)
		img = img[:,:,:3]
		img = img.reshape(1, -1)

		# make prediction on new image
		prediction = model.predict(img)
	
		# squeeze value from 1D array and convert to string for clean return
		label = str(np.squeeze(prediction))

		# switch for case where label=10 and number=0
		if label=='10': label='0'

		return render_template('index.html', label=label)


if __name__ == '__main__':
	# load ml model
	model = joblib.load('model.pkl')
	# start api
	app.run(host='0.0.0.0', port=8000, debug=True)
