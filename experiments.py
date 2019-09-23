import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd

from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


def run_decision_tree_experiment():
    credit_default = pd.read_excel('default_of_credit_card_clients.xls', header=1, index_col=0)
    credit_default.columns = map(str.lower, credit_default.columns)
    credit_default.rename(columns={'default payment next month': 'default_payment'}, inplace=True)
    feature_cols = credit_default.columns.values[:-1]

    X_credit_default = credit_default[feature_cols]
    y_credit_default = credit_default.default_payment
    X_train, X_test, y_train, y_test = train_test_split(X_credit_default, y_credit_default, test_size=.35, random_state=42, stratify=y_credit_default)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_credit_default_test = sc.transform(X_test)
    y_credit_default_test = y_test
    X_credit_default, y_credit_default = X_train.copy(), y_train.copy()

    plot_learning_curve(DecisionTreeClassifier(max_depth=5), X_credit_default, y_credit_default)
    plot_learning_curve_times(DecisionTreeClassifier(max_depth=5), X_credit_default, y_credit_default)
    plot_validation_curves("tree", X_credit_default, y_credit_default)

    credit_default_tree_cf = DecisionTreeClassifier(max_depth=5)
    credit_default_tree_cf.fit(X_credit_default, y_credit_default)
    y_pred = credit_default_tree_cf.predict(X_credit_default_test)
    print("Accuracy:",metrics.accuracy_score(y_credit_default_test, y_pred))

    pen_digits = pd.read_csv('pendigits.csv')

    feature_cols = pen_digits.columns.values[:-1]
    X_pen_digits = pen_digits[feature_cols]
    y_pen_digits = pen_digits["8"]
    X_train, X_test, y_train, y_test = train_test_split(X_pen_digits, y_pen_digits, test_size=.3, random_state=42, stratify=y_pen_digits)
    X_pen_digits_test = X_test
    y_pen_digits_test = y_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_pen_digits, y_pen_digits = X_train.copy(), y_train.copy()

    plot_learning_curve(DecisionTreeClassifier(max_depth=5), X_pen_digits, y_pen_digits)
    plot_learning_curve_times(DecisionTreeClassifier(max_depth=5), X_pen_digits, y_pen_digits)
    plot_validation_curves("tree", X_pen_digits, y_pen_digits)

    pen_digits_tree_cf = DecisionTreeClassifier(max_depth=15)
    pen_digits_tree_cf.fit(X_pen_digits, y_pen_digits)
    y_predict = pen_digits_tree_cf.predict(X_pen_digits_test)
    print("Accuracy:",metrics.accuracy_score(y_pen_digits_test, y_pred))


def run_neural_network_experiment():
    credit_default = pd.read_excel('default_of_credit_card_clients.xls', header=1, index_col=0)
    credit_default.columns = map(str.lower, credit_default.columns)
    credit_default.rename(columns={'default payment next month': 'default_payment'}, inplace=True)
    feature_cols = credit_default.columns.values[:-1]

    X_credit_default = credit_default[feature_cols]
    y_credit_default = credit_default.default_payment
    X_train, X_test, y_train, y_test = train_test_split(X_credit_default, y_credit_default, test_size=.35, random_state=42, stratify=y_credit_default)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_credit_default_test = sc.transform(X_test)
    y_credit_default_test = y_test
    X_credit_default, y_credit_default = X_train.copy(), y_train.copy()

    plot_learning_curve(MLPClassifier(), X_credit_default, y_credit_default)
    plot_learning_curve_times(MLPClassifier(), X_credit_default, y_credit_default)
    plot_validation_curves("neural network", X_credit_default, y_credit_default)

    credit_default_nn_cf = MLPClassifier(hidden_layer_sizes=7, activation="logistic")
    credit_default_nn_cf.fit(X_credit_default, y_credit_default)
    y_pred = credit_default_nn_cf.predict(X_credit_default_test)
    print("Accuracy:",metrics.accuracy_score(y_credit_default_test, y_pred))

    pen_digits = pd.read_csv('pendigits.csv')

    feature_cols = pen_digits.columns.values[:-1]
    X_pen_digits = pen_digits[feature_cols]
    y_pen_digits = pen_digits["8"]
    X_train, X_test, y_train, y_test = train_test_split(X_pen_digits, y_pen_digits, test_size=.3, random_state=42, stratify=y_pen_digits)
    X_pen_digits_test = X_test
    y_pen_digits_test = y_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_pen_digits, y_pen_digits = X_train.copy(), y_train.copy()

    plot_learning_curve(MLPClassifier(), X_pen_digits, y_pen_digits)
    plot_learning_curve_times(MLPClassifier(), X_pen_digits, y_pen_digits)
    plot_validation_curves("neural network", X_pen_digits, y_pen_digits)

    pen_digits_nn_cf = MLPClassifier()
    pen_digits_nn_cf.fit(X_pen_digits, y_pen_digits)
    y_pred = pen_digits_nn_cf.predict(X_pen_digits_test)
    print("Accuracy:",metrics.accuracy_score(y_pen_digits_test, y_pred))


def run_boosting_experiment():
    credit_default = pd.read_excel('default of credit card clients.xls', header=1, index_col=0)
    credit_default.columns = map(str.lower, credit_default.columns)
    credit_default.rename(columns={'default payment next month': 'default_payment'}, inplace=True)
    feature_cols = credit_default.columns.values[:-1]

    X_credit_default = credit_default[feature_cols]
    y_credit_default = credit_default.default_payment
    X_train, X_test, y_train, y_test = train_test_split(X_credit_default, y_credit_default, test_size=.35, random_state=42, stratify=y_credit_default)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_credit_default_test = sc.transform(X_test)
    y_credit_default_test = y_test
    X_credit_default, y_credit_default = X_train.copy(), y_train.copy()

    plot_learning_curve(AdaBoostClassifier(), X_credit_default, y_credit_default)
    plot_learning_curve_times(AdaBoostClassifier(), X_credit_default, y_credit_default)
    plot_validation_curves("boosting", X_credit_default, y_credit_default)

    credit_default_boosting_cf = AdaBoostClassifier(n_estimators=25)
    credit_default_boosting_cf.fit(X_credit_default, y_credit_default)
    y_pred = credit_default_boosting_cf.predict(X_credit_default_test)
    print("Accuracy:",metrics.accuracy_score(y_credit_default_test, y_pred))

    pen_digits = pd.read_csv('pendigits.csv')

    feature_cols = pen_digits.columns.values[:-1]
    X_pen_digits = pen_digits[feature_cols]
    y_pen_digits = pen_digits["8"]
    X_train, X_test, y_train, y_test = train_test_split(X_pen_digits, y_pen_digits, test_size=.3, random_state=42, stratify=y_pen_digits)
    X_pen_digits_test = X_test
    y_pen_digits_test = y_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_pen_digits, y_pen_digits = X_train.copy(), y_train.copy()

    plot_learning_curve(AdaBoostClassifier(), X_pen_digits, y_pen_digits)
    plot_learning_curve_times(AdaBoostClassifier(), X_pen_digits, y_pen_digits)
    plot_validation_curves("boosting", X_pen_digits, y_pen_digits)

    pen_digits_boosting_cf = AdaBoostClassifier()
    pen_digits_boosting_cf.fit(X_pen_digits, y_pen_digits)
    y_pred = pen_digits_boosting_cf.predict(X_pen_digits_test)
    print("Accuracy:",metrics.accuracy_score(y_pen_digits_test, y_pred))


def run_support_vector_machine_experiment():
    credit_default = pd.read_excel('default of credit card clients.xls', header=1, index_col=0)
    credit_default.columns = map(str.lower, credit_default.columns)
    credit_default.rename(columns={'default payment next month': 'default_payment'}, inplace=True)
    feature_cols = credit_default.columns.values[:-1]

    X_credit_default = credit_default[feature_cols]
    y_credit_default = credit_default.default_payment
    X_train, X_test, y_train, y_test = train_test_split(X_credit_default, y_credit_default, test_size=.35, random_state=42, stratify=y_credit_default)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_credit_default_test = sc.transform(X_test)
    y_credit_default_test = y_test
    X_credit_default, y_credit_default = X_train.copy(), y_train.copy()

    plot_learning_curve(SVC(kernel="linear"), X_credit_default, y_credit_default)
    plot_learning_curve_times(SVC(kernel="linear"), X_credit_default, y_credit_default)
    plot_validation_curves("svc", X_credit_default, y_credit_default)

    plot_learning_curve(SVC(kernel="rbf"), X_credit_default, y_credit_default)
    plot_learning_curve_times(SVC(kernel="rbf"), X_credit_default, y_credit_default)
    plot_validation_curves("svc", X_credit_default, y_credit_default)

    credit_default_svm_linear_cf = SVC(kernel='linear')
    credit_default_svm_linear_cf.fit(X_credit_default, y_credit_default)
    y_pred = credit_default_svm_linear_cf.predict(X_credit_default_test)
    print("Accuracy:",metrics.accuracy_score(y_credit_default_test, y_pred))

    credit_default_svm_rbf_cf = SVC(kernel='rbf')
    credit_default_svm_rbf_cf.fit(X_credit_default, y_credit_default)
    y_pred = credit_default_svm_rbf_cf.predict(X_credit_default_test)
    print("Accuracy:",metrics.accuracy_score(y_credit_default_test, y_pred))

    pen_digits = pd.read_csv('pendigits.csv')

    feature_cols = pen_digits.columns.values[:-1]
    X_pen_digits = pen_digits[feature_cols]
    y_pen_digits = pen_digits["8"]
    X_train, X_test, y_train, y_test = train_test_split(X_pen_digits, y_pen_digits, test_size=.3, random_state=42, stratify=y_pen_digits)
    X_pen_digits_test = X_test
    y_pen_digits_test = y_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_pen_digits, y_pen_digits = X_train.copy(), y_train.copy()

    plot_learning_curve(SVC(kernel="linear"), X_pen_digits, y_pen_digits)
    plot_learning_curve_times(SVC(kernel="linear"), X_pen_digits, y_pen_digits)
    plot_validation_curves("svc", X_pen_digits, y_pen_digits)

    plot_learning_curve(SVC(kernel="rbf"), X_pen_digits, y_pen_digits)
    plot_learning_curve_times(SVC(kernel="rbf"), X_pen_digits, y_pen_digits)
    plot_validation_curves("svc", X_pen_digits, y_pen_digits)

    pen_digits_svm_linear_cf = SVC(kernel="linear")
    pen_digits_svm_linear_cf.fit(X_pen_digits, y_pen_digits)
    y_pred = pen_digits_svm_linear_cf.predict(X_pen_digits_test)
    print("Accuracy:",metrics.accuracy_score(y_pen_digits_test, y_pred))
    print(cross_val_score(SVC(kernel="linear"), X_pen_digits, y_pen_digits, cv=3))

    pen_digits_svm_rbf_cf = SVC(kernel="rbf")
    pen_digits_svm_rbf_cf.fit(X_pen_digits, y_pen_digits)
    y_pred = pen_digits_svm_rbf_cf.predict(X_pen_digits_test)
    print("Accuracy:",metrics.accuracy_score(y_pen_digits_test, y_pred))


def run_k_nearest_neighbors_experiment():
    credit_default = pd.read_excel('default of credit card clients.xls', header=1, index_col=0)
    credit_default.columns = map(str.lower, credit_default.columns)
    credit_default.rename(columns={'default payment next month': 'default_payment'}, inplace=True)
    feature_cols = credit_default.columns.values[:-1]

    X_credit_default = credit_default[feature_cols]
    y_credit_default = credit_default.default_payment
    X_train, X_test, y_train, y_test = train_test_split(X_credit_default, y_credit_default, test_size=.35, random_state=42, stratify=y_credit_default)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_credit_default_test = sc.transform(X_test)
    y_credit_default_test = y_test
    X_credit_default, y_credit_default = X_train.copy(), y_train.copy()

    plot_learning_curve(KNeighborsClassifier(), X_credit_default, y_credit_default)
    plot_learning_curve_times(KNeighborsClassifier(), X_credit_default, y_credit_default)
    plot_validation_curves("nearest neighbors", X_credit_default, y_credit_default)

    credit_default_knn_cf = KNeighborsClassifier(n_neighbors=15)
    credit_default_knn_cf.fit(X_credit_default, y_credit_default)
    y_pred = credit_default_knn_cf.predict(X_credit_default_test)
    print("Accuracy:",metrics.accuracy_score(y_credit_default_test, y_pred))

    pen_digits = pd.read_csv('pendigits.csv')

    feature_cols = pen_digits.columns.values[:-1]
    X_pen_digits = pen_digits[feature_cols]
    y_pen_digits = pen_digits["8"]
    X_train, X_test, y_train, y_test = train_test_split(X_pen_digits, y_pen_digits, test_size=.3, random_state=42, stratify=y_pen_digits)
    X_pen_digits_test = X_test
    y_pen_digits_test = y_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_pen_digits, y_pen_digits = X_train.copy(), y_train.copy()

    plot_learning_curve(KNeighborsClassifier(), X_pen_digits, y_pen_digits)
    plot_learning_curve_times(KNeighborsClassifier(), X_pen_digits, y_pen_digits)
    plot_validation_curves("nearest neighbors", X_pen_digits, y_pen_digits)


def plot_learning_curve(classifier, X, y, cv=3, scoring='accuracy',
                        n_jobs=-1, training_sizes=np.linspace(0.01, 1.0, 50)):
    train_sizes, train_scores, test_scores, _, _ = learning_curve(classifier,
                                                                  X=X,
                                                                  y=y,
                                                                  # Number of folds in cross-validation
                                                                  cv=cv,
                                                                  # Evaluation metric
                                                                  scoring=scoring,
                                                                  # Use all computer cores
                                                                  n_jobs=n_jobs,
                                                                  # 50 different sizes of the training set
                                                                  train_sizes=training_sizes,
                                                                  return_times=True)

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.show()


def plot_learning_curve_times(classifier, X, y, cv=3, scoring='accuracy',
                              n_jobs=-1, training_sizes=np.linspace(0.01, 1.0, 50)):
    train_sizes, train_scores, fit_times, score_times, fit_times = learning_curve(classifier,
                                                                                  X=X,
                                                                                  y=y,
                                                                                  # Number of folds in cross-validation
                                                                                  cv=cv,
                                                                                  # Evaluation metric
                                                                                  scoring=scoring,
                                                                                  # Use all computer cores
                                                                                  n_jobs=n_jobs,
                                                                                  # 50 different sizes of the training set
                                                                                  train_sizes=training_sizes,
                                                                                  return_times=True)

    # Create means and standard deviations of training set scores
    fit_mean = np.mean(fit_times, axis=1)
    fit_std = np.std(fit_times, axis=1)

    # Create means and standard deviations of test set scores
    score_mean = np.mean(score_times, axis=1)
    score_std = np.std(score_times, axis=1)

    # Draw lines
    plt.plot(train_sizes, fit_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, score_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, fit_mean - fit_std, fit_mean + fit_std, color="#DDDDDD")
    plt.fill_between(train_sizes, score_mean - score_std, score_mean + score_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Time"), plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("svc_rbf_learning_curve_times.png")
    plt.show()


def plot_validation_curves(classifier_name, X, y):
    if classifier_name == "tree":
        plot_decision_tree_validation_curves(DecisionTreeClassifier(), X, y)
    elif classifier_name == "neural network":
        plot_neural_network_validation_curves(MLPClassifier(), X, y)
    elif classifier_name == "nearest neighbors":
        plot_nearest_neighbor_validation_curves(KNeighborsClassifier(), X, y)
    elif classifier_name == "boosting":
        plot_boosting_validation_curves(AdaBoostClassifier(), X, y)
    elif classifier_name == "svc":
        plot_support_vector_machine_validation_curves(SVC(), X, y)


def plot_decision_tree_validation_curves(estimator, X, y):
    plot_max_depth_validation_curve(estimator, X, y)
    plot_criterion_validation_curve(estimator, X, y)


def plot_neural_network_validation_curves(estimator, X, y):
    plot_alpha_validation_curve(estimator, X, y)
    plot_activiation_validation_curve(estimator, X, y)
    plot_nn_max_iter_validation_curve(estimator, X, y)


def plot_nearest_neighbor_validation_curves(estimator, X, y):
    plot_k_validation_curve(estimator, X, y)
    plot_weights_validation_curve(estimator, X, y)


def plot_boosting_validation_curves(estimator, X, y):
    plot_estimator_validation_curve(estimator, X, y)
    plot_learning_rate_validation_curve(estimator, X, y)


def plot_support_vector_machine_validation_curves(estimator, X, y):
    plot_svm_max_iter_validation_curve(estimator, X, y)
    plot_penalty_parameter_validation_curve(estimator, X, y)


def plot_max_depth_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = np.arange(1, 100, 5)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="max_depth",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With Decision Tree")
    plt.xlabel("Depth Of Tree")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("tree_mc_depth.png")
    plt.show()


def plot_criterion_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = ["gini", "entropy"]

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="criterion",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With Decision Tree")
    plt.xlabel("Estimator")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("tree_criterion_validation_curve.png")
    plt.show()


def plot_nn_max_iter_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = np.arange(200, 1000, 200)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="max_iter",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With Neural Network")
    plt.xlabel("Max Iterations")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("nn_max_iter_validation_curve.png")
    plt.show()


def plot_alpha_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = np.array([.00001, .0001, .001, .01])

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="alpha",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With Neural Network")
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("nn_alpha_validation_curve.png")
    plt.show()


def plot_activiation_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = ["identity", "logistic", "tanh", "relu"]

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="activation",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With Neural Network")
    plt.xlabel("Activation")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("nn_activation_validation_curve.png")
    plt.show()


def plot_k_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = np.arange(5, 25, 5)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="n_neighbors",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With K Nearest Neighbors")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("knn_neighbhors_validation_curve.png")
    plt.show()


def plot_weights_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = ["uniform", "distance"]

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="weights",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With K Nearest Neighbors")
    plt.xlabel("Weights")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("knn_weights_validation_curve.png")
    plt.show()


def  plot_estimator_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = np.arange(20, 70, 5)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="n_estimators",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With Boosting")
    plt.xlabel("Estimators")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("boosting_n_estimator_validation_curve.png")
    plt.show()


def plot_learning_rate_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = np.arange(1, 4, 2)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="learning_rate",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With Boosting")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("boosting_learning_rate_validation_curve.png")
    plt.show()


def plot_svm_max_iter_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = np.arange(1000, 2000, 200)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="max_iter",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With SVM")
    plt.xlabel("Max Iterations")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("svc_max_iter_validation_curve.png")
    plt.show()


def plot_penalty_parameter_validation_curve(estimator, X, y):
    # Create range of values for parameter
    param_range = np.arange(2, 10, 2)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator,
                                                 X,
                                                 y,
                                                 param_name="C",
                                                 param_range=param_range,
                                                 cv=3,
                                                 scoring="accuracy",
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With SVM")
    plt.xlabel("Penalty")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("svc_penalty_parameter_validation_curve.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='this will be the controller for the experiments')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    parser.add_argument('-t', help='decision tree classifier', action='store_true')
    parser.add_argument('-nn', help='neural network classifier', action='store_true')
    parser.add_argument('-b', help='boosting ensemble classifier', action='store_true')
    parser.add_argument('-svm', help='support vector machine classifier', action='store_true')
    parser.add_argument('-knn', help='k nearest neighbor classifier', action='store_true')
    parser.add_argument('-all', help='run all classifiers', action='store_true')
    args = parser.parse_args()

    if args.t or args.all:
        run_decision_tree_experiment()
    if args.nn or args.all:
        run_neural_network_experiment()
    if args.b or args.all:
        run_boosting_experiment()
    if args.svm or args.all:
        run_support_vector_machine_experiment()
    if args.knn or args.all:
        run_k_nearest_neighbors_experiment()




