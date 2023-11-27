import csv
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, r_regression
from sklearn.metrics import accuracy_score

dataset = 'eeg'
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1410),

}
preprocs = {
    'none': None,
    'ros': RandomOverSampler(random_state=1410),
    'smote' : SMOTE(random_state=1410),
    'rus': RandomUnderSampler(random_state=1410),
}
metrics = {
    "recall": recall_score,
    'precision': precision_score,
    'f1': f1_score,
    'bac': balanced_accuracy_score,
}
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)


def experiment_0(preproc, preproc_id):
    scores2 = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            X_train, y_train = preprocs[preproc].fit_resample(X[train], y[train])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X[test])
            scores2[clf_id, fold_id] = accuracy_score(y[test], y_pred)
    np.save(f"results_{preproc_id}", scores2)

def experiment_1(clf_key, clf, preprocs, metrics, X, y, rskf):
    scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for preproc_id, preproc in enumerate(preprocs):
            clf = clone(clf)
            if preprocs[preproc] == None:
                X_train, y_train = X[train], y[train]
            else:
                X_train, y_train = preprocs[preproc].fit_resample(X[train], y[train])

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X[test])
            for metric_id, metric in enumerate(metrics):
                scores[preproc_id, fold_id, metric_id] = metrics[metric](y[test], y_pred)
    np.save(f"results_{clf_key}", scores)


clfs_preproc = {
    'GNB-RUS': (GaussianNB(), RandomUnderSampler(random_state=1410)),
    'kNN-SMOTE': (KNeighborsClassifier(), SMOTE(random_state=1410)),
    'CART-ROS': (DecisionTreeClassifier(random_state=1410), RandomOverSampler(random_state=1410))
}

def experiment2():
    scores = {'k_best': {'GNB-RUS': {}, 'kNN-SMOTE': {}, 'CART-ROS': {}}, 
              'lasso': {'GNB-RUS': {}, 'kNN-SMOTE': {}, 'CART-ROS': {}}, 
              'recursive': {'GNB-RUS': {}, 'kNN-SMOTE': {}, 'CART-ROS': {}}}
    for k in [4, 6, 8, 10, 12, 14]:
    # for k in [4,12, 14]:
        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for i, (test_id, (clf, preproc)) in enumerate(clfs_preproc.items()):
                clf = clone(clf)
                X_train, y_train = preproc.fit_resample(X[train], y[train])
                
                # k_best = SelectKBest(r_regression, k=k)
                # X_train_selected = k_best.fit_transform(X_train, y_train)
                # X_test_selected = k_best.transform(X[test])
                # clf.fit(X_train_selected, y_train)
                # y_pred = clf.predict(X_test_selected)
                # accuracy = accuracy_score(y[test], y_pred)
                # scores['k_best'][test_id][k] = accuracy

                # lasso = SelectFromModel(LassoCV(tol=1e-1, max_iter=30000), max_features=k)
                # X_train_selected = lasso.fit_transform(X_train, y_train)
                # X_test_selected = lasso.transform(X[test])
                # clf.fit(X_train_selected, y_train)
                # y_pred = clf.predict(X_test_selected)
                # accuracy = accuracy_score(y[test], y_pred)
                # scores[f'{test_id}_{k}_lasso'] = accuracy

                estimator = SVR(kernel="linear")
                recursive = RFE(estimator= estimator, n_features_to_select=k)
                X_train_selected = recursive.fit_transform(X_train, y_train)
                X_test_selected = recursive.transform(X[test])
                clf.fit(X_train_selected, y_train)
                y_pred = clf.predict(X_test_selected)
                accuracy = accuracy_score(y[test], y_pred)
                scores['recursive'][test_id][k] = accuracy
                print(k)

        # mean_score = np.mean(scores)
        # std_score = np.std(scores)
        # print(f"Accuracy score for {k} features: %.3f (%.3f)" % (mean_score, std_score))
    print(scores)


    k_values = list(scores['recursive']['GNB-RUS'].keys())
    for classifier, values in scores['recursive'].items():
        plt.plot(k_values, list(values.values()), label=classifier)
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('recursive', dpi=200)
    plt.show()

if __name__=='__main__':
    # for clf_key in clfs:
    #     experiment_1(clf_key, clfs[clf_key], preprocs, metrics, X, y, rskf)
    
    #experiment2()
    for preproc_id, preproc in enumerate(preprocs):
        experiment_0(preproc, preproc_id)

