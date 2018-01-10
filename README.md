Source code for the tutorial 'Deploying a machine learning model with a Flask API' written for [HyperionDev](http://blog.hyperiondev.com).

In this tutorial we take the image classification model built in `model.py` which recognises Google Street View House Numbers. Using Flask to create an API, we can deploy this model and create a simple web page to load and classify new images. 

To run locally:

- Install pip and Python 3
- Clone this repository `git clone https://github.com/elliebirbeck/model-deployment-flask.git`
- Navigate to the working directory `cd model-deployment-flask`
- Install the Python dependencies `pip install -r requirements.txt`
- Run the API `python api.py`
- Open a web browser and go to `http://localhost:8000`

![screenshot.png](screenshot.png)