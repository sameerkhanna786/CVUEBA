import pandas as pd
import numpy as np
import pickle

"""
This script assumes that the CERT dataset is downloaded and saved to location "..".
Extracts non-dynamic information regarding all users and stores within a dictionary for easy access later on.
"""

infoDict = {}

psycho = pd.read_csv("../Dataset/r4.2/psychometric.csv")
psycho = psycho.drop("employee_name", axis=1)

file_dir = "../Dataset/r4.2/LDAP"
id_to_role = {}

#List of which files in the r4.2 subdirectory to search through.
csv_lst = [
	'2010-01',
	'2010-02',
	'2010-03',
	'2010-07',
	'2010-12',
	'2010-06',
	'2010-10',
	'2010-04',
	'2010-05', 
	'2010-11', 
	'2010-08', 
	'2010-09', 
	'2009-12', 
	'2011-03', 
	'2011-02', 
	'2011-01', 
	'2011-05', 
	'2011-04'
]

for csv in csv_lst:
	data = pd.read_csv(file_dir + "/" + csv + ".csv")
	users = list(data.user_id.unique())
	for user in users:
		cur = data[data.user_id == user]
		role_val = list(cur["role"])[0]
		id_to_role[user] = role_val

psycho["role"] = psycho["user_id"].map(id_to_role)
psycho = pd.get_dummies(psycho, prefix='role', columns=['role'])
user_lst = list(psycho.user_id.unique())

for user in user_lst:
	cur_psycho = psycho[psycho.user_id == user]
	cur_psycho = np.squeeze(np.array(cur_psycho.drop("user_id", axis=1)))
	infoDict[user] = cur_psycho

pickle.dump(infoDict, open("nondynamic.pkl","wb"))
