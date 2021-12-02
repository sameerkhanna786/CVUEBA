# Computer Vision User Entity Behavior Analytics
Code repository for [Computer Vision User Entity Behavior Analytics](https://arxiv.org/abs/2111.13176)

## Code Description

### dataset.csv
As discussed in the manuscript, CVUEBA was designed to be utilized in production. Thus, as an extra layer of security, we keep the features used as well as the feature extraction module proprietary. 

We observed that one can obtain similar performance on the CERT Insider Threat dataset using a combination of features introduced by various publications in concert with the features we introduce in the main manuscript.

dataset.csv is a CSV file containing the extracted features for various users for various days in the CERT Insider Threat dataset. For space reasons, we publish a small segment of the original dataset here. Reported instances were chosen by randomly selecting from the set of encoded images used to evaluate CVUEBA and storing unique behavior instances corresponding to the channels of these images.

We did not wish for all of the code to be proprietary, and thus felt this was an acceptable compromise.

## split_dataset.py
Splits dataset into train, test, and validation sets.

## sae_hopt.py & SAE.hyperopt
This script is used for hyperparameter search for the SAE model using the HyperOpt module. Results of tuning are stored within SAE.hyperopt.

## SAE.py
Defines the SAE model. Optimal hyperparameters are determined as shown in the script sae_hopt.py.

## generate_images.py
Trains the SAE model using optimal parameters stored in SAE.hyperopt if a trained model is not present. Uses this model to generate color image encodings of behavior.

## extract_non_dynamic.py and nondynamic.pkl
CVUEBA uses non-dynamic information to improve model precision. This script extracts the information from the CERT Insider Threat dataset and stores it within nondynamic.pkl.

To execute this script you would need to download the CERT Insider Threat dataset. For demo purposes, we provide a pre-extracted pickle file in the repo.

## prep_data_model.py
This is a custom data loader that uses the image directory name and nondynamic.pkl to pull the information to be passed into the CVUEBA model.

## CVUEBA.py
Loads train and test set data, builds CVUEBA model, trains and saves model, and reports evaluation metrics.

## How To Use
We provide a requirements.txt file that lists all dependencies required to run the demo.

The script run.sh is provided to execute all the various python scripts in order to split data, generate images, and evaluate CVUEBA.
