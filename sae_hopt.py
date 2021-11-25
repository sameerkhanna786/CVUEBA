from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import math
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
import pickle
from sklearn.metrics import balanced_accuracy_score
import traceback
from SAE import *
from sklearn.linear_model import SGDClassifier

"""
Tune SAE model using the Tree Parzen Estimator.

Uses the HyperOpt API:
http://hyperopt.github.io/hyperopt/
"""

#Load training data
data = pd.read_csv("train.csv")
data = data.drop(["user", "day"], axis=1)

X = data.drop("attack", axis=1)
y = data["attack"]
y_train[y_train!=0] = 1

#We train on only negative class.
X_train = X[y_train==0]

dev = pd.read_csv("valid.csv")
dev = dev.drop(["user", "day"], axis=1)

X_dev = dev.drop("attack", axis=1)
y_dev = dev["attack"]
y_dev[y_dev!=0] = 1

def objective(args):
    try:
        model = SAE(**args)

        model.fit(X_train)

        #Use SVM to eval separability
        svm = SGDClassifier(n_jobs=-1, class_weight='balanced')
        svm.fit(model.predict(X_train), y_train)

        y_pred = svm.predict(model.predict(X_dev))

        val = -balanced_accuracy_score(y_dev, y_pred)

        if math.isnan(val) or val is None:
            return {'loss': float('inf'), 'status': STATUS_FAIL }
        return {'loss': val, 'status': STATUS_OK }

    except Exception as e:
        print(e)
        traceback.print_exc()
        return {'loss': float('inf'), 'status': STATUS_FAIL }

# define a search space
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

_model = "SAE"

def run_trials():
    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 20  # initial max_trials. put something small to not have to wait
    try:  # try to load an already saved trials object, and increase the max
        with open(f"{_model}.hyperopt", "rb") as f:
            trials = pickle.load(f)
            print("Found saved Trials! Loading...")
            max_trials = len(trials.trials) + trials_step
            print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))

    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_trials, trials=trials)
    print("Best:", space_eval(space, best))

    # save the trials object
    with open(f"{_model}.hyperopt", "wb") as f:
        pickle.dump(trials, f)

# loop indefinitely and stop whenever you like
while True:
    run_trials()
