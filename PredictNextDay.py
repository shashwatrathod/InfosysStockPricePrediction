import tensorflow as tf
from tensorflow.keras.models import Sequential
import pandas_datareader as web
from datetime import datetime
import pandas as pd
import numpy as np
import pickle

def inverse_transform(X):
  with open("normalization_factors.pkl","rb") as f:
    norm_factors = pickle.load(f)
  
  X = X*norm_factors
  return X

def transform(X):
  with open("normalization_factors.pkl","rb") as f:
    norm_factors = pickle.load(f)
  
  return X/norm_factors

def predict_next_date():
  model = tf.keras.models.load_model('models/STOCKPRED-1582435245')  
  today = datetime.today().strftime('%d-%m-%y')
  df = web.DataReader('INFY',data_source='yahoo',start='20-02-2016', end=today)
  closing_data = df['Close']
  test_next = np.asarray(closing_data[-90:])
  test_next = transform(test_next)
  test_n = []
  test_n.append(test_next[0:90])
  test_n = np.asarray(test_n)
  test_n = np.reshape(test_n,(test_n.shape[0],test_n.shape[1],1))
  pred_n = model.predict(test_n)
  pred_inv = inverse_transform(pred_n)
  print(f'Next date\'s prediction: USD {pred_inv[0,0]}')

predict_next_date()