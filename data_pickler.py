import csv
import numpy as np
import pickle
import sklearn.preprocessing

from sklearn.model_selection import train_test_split

pickle_file_name = './data/onehot_data.pkl'
csv_file_name = './data/input.csv'
write_file_name = './data/full_data.pkl'

picklefile = open(pickle_file_name, 'rb')
csvfile = open(csv_file_name)
writefile = open(write_file_name, 'wb+')

reader = csv.DictReader(csvfile)
sets = pickle.load(picklefile)

data = [] # [num_samples, num_feats (without oenhots)]
onehot_data = [] # [num_smaples, num_onehot_feats]
labels = [] # [num_samples]

def make_onehot(label, total):
    zeros = np.zeros(total)
    zeros[label] = 1
    return zeros

features = ['Total_Assets', 'Total_Liabilities', 'Total_Net_Assets', 'Wkg_Capital',
        'Total_Contributions', 'ProgSvcRev', 'Other_Revenue', 'Total_Revenue', 'Fundraising_Expenses',
        'Administration_Expenses', 'Program_Expenses', 'Total_Expenses']

lineno = 1
for line in reader:
    try:
        labels.append(int(line['OverallRtg']))
    except ValueError:
        labels.append(-1)

    feats = [float(line[item]) for item in features]
    data.append(feats)

    onehot_feats = []
    for key in sets.keys():
        label = sets[key][line[key]]
        total = sets[key]['total']
        onehot_feats.append(make_onehot(label, total))
    onehot_feats = np.concatenate(onehot_feats, axis=0)
    onehot_data.append(onehot_feats)

    lineno += 1

data, onehot_data, labels = [np.array(arr) for arr in 
    [data, onehot_data, labels]]

#for label in [-1, 0, 1, 2, 3, 4]:
#    print(np.sum(labels==label) / float(len(labels)))

data = np.concatenate((data, onehot_data), axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)

for label in [-1, 0, 1, 2, 3, 4]:
    print(np.sum(y_train==label) / float(len(y_train)))

data_train, data_test = [arr[:, 0:len(features)] for arr in [X_train, X_test]]
scaler = sklearn.preprocessing.StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

X_train[:, 0:len(features)] = data_train
X_test[:, 0:len(features)] = data_test

pickle.dump([X_train, X_test, y_train, y_test], writefile)

picklefile.close()
csvfile.close()
writefile.close()
