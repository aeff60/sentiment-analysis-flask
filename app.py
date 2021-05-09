#from PIL import Image, ImageDraw
#import face_recognition
#import numpy as np
#import pickle

#import io
#from datetime import datetime
from flask import Flask, request
from flask_cors import CORS
from nltk import NaiveBayesClassifier as nbc
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
import joblib
import pickle

# Load face names and encoding from pickle


mod = joblib.load('sentiment.model')
mod2 = joblib.load('vocabulary.model2')

app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return 'Hello'

@app.route('/api/predict_sentiment/', methods = ['GET'])
def predict_sentiment():
    text = request.values['param']
    test_sentence = str(text)
    featurized_test_sentence = {
        i: (i in word_tokenize(test_sentence.lower())) for i in mod2}
    classifier_2 = mod.classify(featurized_test_sentence)
    return {'results': classifier_2}


if __name__== '__main__':
    app.run(threaded=True)      