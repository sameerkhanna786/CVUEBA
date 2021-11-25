import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import *
from sklearn.model_selection import train_test_split
import numpy as np
import ssl
from sklearn.metrics import *
from prep_data_model import *

"""
Load or Train dual-input classifier.
Use the data loading API defined in prep_data_model
to extract data
"""

# Need to download ResNet weight (trained on ImageNet)
# This circumvents potential download issues of weights.
ssl._create_default_https_context = ssl._create_unverified_context

print("Loading data...")
images, non_behave, y = load_data("AugImages")
y[y != 0] = 1

print("...Done")

res_model = ResNet50(include_top = False, pooling = "avg", input_shape=(32, 32, 3))
for i, layer in enumerate(res_model.layers):
	if i <= 170:
		layer.trainable = False
	print(f"Layer {i} Out: {layer} Trainable: {layer.trainable}")

try:
	model = tf.keras.models.load_model("cvueba")
	print("Finished Loading Model!")
except:
	print("Model not found. Generating and training!")

	image_input = Input(shape=(32, 32, 3))
	vector_input = Input(shape=(47,))
	res_out = res_model(image_input)

	concat_layer = Concatenate()([vector_input, res_out])

	out = Dense(1024, activation='relu', kernel_regularizer="l2")(concat_layer)
	out = Dense(512, activation='relu', kernel_regularizer="l2")(out)
	out = Dense(256, activation='relu', kernel_regularizer="l2")(out)
	out = Dense(1, activation='sigmoid')(out)

	model = Model(inputs=[image_input, vector_input], outputs=out)

	METRICS = [
		tf.keras.metrics.BinaryAccuracy(name='accuracy'),
		tf.keras.metrics.Precision(name='precision'),
		tf.keras.metrics.Recall(name='recall'),
	]

	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=METRICS)
	model.fit([images, non_behave], y, batch_size=128, epochs=100, shuffle=True)

	model.save("cvueba")

test_images, test_non_behave, y_test = load_data("TestImages")
y_test[y_test!=0] = 1

print("Predicting...")
y_pred = model.predict([test_images, test_non_behave])
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1
print("...Finished!")

bal_accuracy = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Balanced Accuracy: {bal_accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
