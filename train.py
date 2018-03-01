import argparse
import pandas
import numpy as np
import pickle

from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser(description='Train SVM model')
parser.add_argument('--type', action='store', type=str)
args = parser.parse_args()

_type = args.type

DATA_FILE = './data/full_data.pkl'
OUTPUT_MODEL_FILE = './models/svm/svm_new.pkl'
OUTPUT_CROSSVAL_FILE = './models/svm/svm_crossval_new.csv'
OUTPUT_TEST_FILE = './models/svm/svm_test_new.txt'

readfile = open(DATA_FILE, 'rb')

X_train, X_test, y_train, y_test = pickle.load(readfile)

if _type == 'train':
    parameter_grid = {'C': [2**i for i in range(-10, 6)],
                      'gamma': [2**i for i in range(-15, 7)]}
    class_weight = {0: 217.17,
        1: 40.72,
        2: 6.682,
        3: 1.249,
        4: 1
    }
    clf = GridSearchCV(estimator=svm.SVC(class_weight=class_weight),
                       param_grid=parameter_grid,
                       cv=5)
    clf.fit(X_train, y_train)
    print(clf.best_score_)
    print(clf.best_params_)
    print(clf.cv_results_)

    cv_results = pandas.DataFrame.from_dict(clf.cv_results_)
    cv_results.to_csv(OUTPUT_CROSSVAL_FILE)

    joblib.dump(clf.best_estimator_, OUTPUT_MODEL_FILE)

if _type == 'test':
    clf = joblib.load(OUTPUT_MODEL_FILE)
    predict_labels = clf.predict(X_test)
    print(np.sum(predict_labels != y_test)/ float(len(predict_labels)))
    for label in [-1, 0, 1, 2, 3, 4]:
        print(np.sum(predict_labels==label) / float(len(predict_labels)))
    testfile = open(OUTPUT_TEST_FILE, 'w+')
    testfile.write(str(np.sum(predict_labels != y_test)))

readfile.close()
