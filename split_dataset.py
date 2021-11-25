import pandas as pd
from sklearn.model_selection import train_test_split
import random
import math

data = pd.read_csv("dataset.csv")
users = list(data.user.unique())
norm_users = []
attack_one_users = []
attack_two_users = []
attack_three_users = []
for user in users:
	user_data = data[data.user.str.contains(user)]
	scenario = max(user_data["attack"])
	if scenario != 0:
		if scenario == 1:
			attack_one_users.append(user)
		elif scenario == 2:
			attack_two_users.append(user)
		else:
			attack_three_users.append(user)
	else:
		norm_users.append(user)

# Shuffle user lists to ensure randomness in dataset split
random.shuffle(norm_users)
random.shuffle(attack_one_users)
random.shuffle(attack_two_users)
random.shuffle(attack_three_users)

# Define split behavior
train_split = 0.7
valid_split = 0.1
test_split = 0.2
assert(train_split + valid_split + test_split == 1.0)

# Function that returns three lists containing users.
# List sizes determined based on split value defined above.
def split_users(user_lst):
	num_users = len(user_lst)
	valid_num = math.ceil(len(user_lst)*valid_split)
	test_num = math.ceil(len(user_lst)*test_split)
	train_num = num_users - valid_num - test_num
	return user_lst[:train_num], user_lst[train_num:(train_num+valid_num)], user_lst[(train_num+valid_num):]

train_lst = []
valid_lst = []
test_lst = []

for user_lst in [norm_users, attack_one_users, attack_two_users, attack_three_users]:
	tr, v, t = split_users(user_lst)
	train_lst.extend(tr)
	valid_lst.extend(v)
	test_lst.extend(t)

train_data = data.loc[data['user'].isin(train_lst)]
valid_data = data.loc[data['user'].isin(valid_lst)]
test_data = data.loc[data['user'].isin(test_lst)]

train_data.to_csv('train.csv', index=False)
valid_data.to_csv('valid.csv', index=False)
test_data.to_csv('test.csv', index=False)
