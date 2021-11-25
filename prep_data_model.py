import numpy as np
import cv2
import os
import pickle

non_dynamic_dict = pickle.load(open("nondynamic.pkl","rb"))

"""
Load data based on image folder.
Use the naming and filing convention for images
defined in generate_images to extract information
like user info and date of user.
Take this information to pull the relevant
non-dynamic information from the pickled dictionary.
"""
def load_data(image_folder):
	images = []
	non_behavior_info = []
	y = []
	for scenario in [0, 1, 2, 3]:
		folder = os.path.join(image_folder, str(scenario))
		for filename in os.listdir(folder):
			path = os.path.join(folder, str(filename))
			user = filename.split("_")[0]
			cur_im = cv2.imread(path)
			non_behavior_info.append(non_dynamic_dict[user])
			images.append(cur_im)
			y.append(scenario)
	return np.stack(images, axis=0), np.array(non_behavior_info), np.array(y)