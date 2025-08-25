import pickle
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)

coefs_has = np.load(r'C:\Users\George\pythonProject\coefs_has.npy')
coefs_no = np.load(r'C:\Users\George\pythonProject\coefs_no.npy')

freqs = np.array([28800, 14400, 9600, 7200, 5760, 4800, 4114.28571429, 3600, 3200])

all = np.vstack((coefs_has, coefs_no))
X = np.empty((40000, 9))

#auto pou allaksa
all = np.absolute(all)
for i in range(len(all)):
    X[i, :] = all[i, :].max(axis=0) - all[i, :].min(axis=0)
    y=np.concatenate((np.ones(len(coefs_has)), np.zeros(len(coefs_no))))

X_has = X[:20000, :]
X_no = X[20000:, :]

X_train = np.vstack((X_has[:15000, :], X_no[:15000, :]))
y_train = np.concatenate((np.ones(15000), np.zeros(15000)))

X_val_has = X_has[15000:17500, :]
X_val_no = X_no[15000:17500, :]
y_val_has = np.ones(2500)
y_val_no = np.zeros(2500)

X_test_has = X_has[17500:, :]
X_test_no = X_no[17500:, :]
y_test_has = np.ones(2500)
y_test_no = np.zeros(2500)

X_val_all = np.vstack((X_val_has, X_val_no))
X_test_all = np.vstack((X_test_has, X_test_no))
y_val_all = np.concatenate((y_val_has, y_val_no))
y_test_all = np.concatenate((y_test_has, y_test_no))


with open('train_only.p', 'wb') as f:
    pickle.dump((X_train, y_train), f)

with open('train_val_all.p', 'wb') as f:
    pickle.dump((X_train, X_val_all, y_train, y_val_all), f)
with open('test_all.p', 'wb') as f:
    pickle.dump((X_test_all, y_test_all), f)

with open('val_has.p', 'wb') as f:
    pickle.dump((X_val_has, y_val_has), f)
with open('val_no.p', 'wb') as f:
    pickle.dump((X_val_no, y_val_no), f)
with open('test_has.p', 'wb') as f:
    pickle.dump((X_test_has, y_test_has), f)
with open('test_no.p', 'wb') as f:
    pickle.dump((X_test_no, y_test_no), f)
with open('X_has.p', 'wb') as f:
    pickle.dump(X_has, f)
with open('X_no.p', 'wb') as f:
    pickle.dump(X_no, f)
with open('test_val_all.p', 'wb') as f:
    pickle.dump((X_test_all, X_val_all, y_test_all, y_val_all), f)


