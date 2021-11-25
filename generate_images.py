def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import pandas as pd
import numpy as np
import math
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from SAE import *
import pickle
from sklearn.preprocessing import MinMaxScaler
import cv2
from tqdm import tqdm

"""
If optimal SAE trained before, load model and scaler and return.
Otherwise, load optimal HyperOpt hyperparameters, train SAE on negative
samples from training set, fit MinMaxScaler and return.
"""
def load_and_train():
	train_data = pd.read_csv("train.csv")
	X = train_data.drop(["user", "day", "attack"], axis=1)
	y = train_data["attack"]
	y[y!=0] = 1
	#We train on only negative class.
	X_train = X[y==0]

	space = {
	    'rho': hp.loguniform('rho', np.log(0.0001), np.log(1)),
	    'beta': hp.loguniform('beta', np.log(0.0001), np.log(1)),
	    'activation': hp.choice('activation', ['relu', 'selu', 'sigmoid', 'tanh', 'elu', 'gelu', 'softsign', 'swish', 'hard_sigmoid']),
	    'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'adamax']),
	    'batch_size': hp.choice('batch_size', [i for i in range(32, 1025)]),
	    'patience': hp.choice('patience', [i for i in range(5, 101)]),
	    'scale': hp.choice('scale', [True, False]),
	    'lambda1': hp.loguniform('lambda1', np.log(0.0001), np.log(1)),
	    'lambda2': hp.loguniform('lambda2', np.log(0.0001), np.log(1)),
	    'verbose': 0,
	    'save': False,
	    'retrain': True,
	}
	trials = pickle.load(open("SAE.hyperopt","rb"))
	best_trial = {}
	for key in trials.best_trial['misc']['vals']:
	    try:
	        best_trial[key] = trials.best_trial['misc']['vals'][key][0]
	    except:
	        best_trial[key] = 0
	argDict = space_eval(space, best_trial)
	argDict["verbose"] = 1
	argDict["verbose"] = 1
	argDict["retrain"] = False
	argDict["save"] = True
	model = SAE(**argDict)

	try:
		model.load()
	except:
		model.fit(X_train)

	if os.path.isfile("scaler.pkl"):
		scaler = pickle.load(open("scaler.pkl","rb"))
	else:
		scaler = MinMaxScaler(feature_range=(0, 255))
		X_trans = model.predict(X_train)
		scaler.fit(X_trans)
		with open("scaler.pkl", "wb") as f:
			pickle.dump(scaler, f)

	return model, scaler

"""
Given a SAE model, and a MinMaxScaler, convert the features stored within
a dataset file into behavior image encodings to be stored in the designated
folder.
"""
def generate_images(csv_file, image_folder, sae_model, min_max_scaler):
	data = pd.read_csv(csv_file)
	data['date'] = pd.to_datetime(data.day, format='%m/%d/%Y', errors='ignore')

	try:
		os.makedirs(f"{image_folder}/0")
	except:
		pass

	try:
		os.makedirs(f"{image_folder}/1")
	except:
		pass

	try:
		os.makedirs(f"{image_folder}/2")
	except:
		pass

	try:
		os.makedirs(f"{image_folder}/3")
	except:
		pass

	users = list(data.user.unique())

	for user in tqdm(users):
		user_data = data[data.user.str.contains(user)]
		dates = list(user_data.date.unique())
		for date in dates:
			cur_date = date
			prev_date = user_data[user_data['date'] < cur_date]['date'].max()
			prev_prev_date = user_data[user_data['date'] < prev_date]['date'].max()

			cur_row = np.array(user_data[user_data['date'] == cur_date].drop(["user", "day", "date", "attack"], axis=1).iloc[0])
			
			if not pd.isnull(prev_prev_date):
				prev_prev_row = np.array(user_data[user_data['date'] == prev_prev_date].drop(["user", "day", "date", "attack"], axis=1).iloc[0])
			else:
				continue

			if not pd.isnull(prev_date):
				prev_row = np.array(user_data[user_data['date'] == prev_date].drop(["user", "day", "date", "attack"], axis=1).iloc[0])
			else:
				continue

			day_str = list(user_data[user_data['date'] == cur_date]["day"])[0].replace("/", "_")
			scenario = list(user_data[user_data['date'] == cur_date]['attack'])[0]

			stack = np.vstack((prev_prev_row, prev_row, cur_row))

			encoded = sae_model.predict(stack)

			encoded_trans = min_max_scaler.transform(encoded)
			
			image = np.reshape(encoded_trans, (3, 32, -1))
			image = np.swapaxes(image, 0, 2)
			image = np.rint(image)

			filename = os.path.join(f"{image_folder}/{scenario}", f"{user}_{day_str}.jpg")
			cv2.imwrite(filename, image)

"""
Trains/Loads SAE model, and generates image encodings for users in
the CERT Insider Threat dataset.
"""
def main():
	sae_model, min_max_scaler = load_and_train()
	generate_images("train.csv", "TrainImages", sae_model, min_max_scaler)
	generate_images("valid.csv", "ValidImages", sae_model, min_max_scaler)
	generate_images("test.csv", "TestImages", sae_model, min_max_scaler)

if __name__ == "__main__":
	main()
