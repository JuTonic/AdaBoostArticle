import time

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import GOMO

df = pd.read_csv("./data/heart_disease.csv")

y = df["HeartDiseaseorAttack"]
X = df.drop(["HeartDiseaseorAttack"], axis=1)

N_ESTIM = 50

############################################
# number of estimators

params = (
    [1, "darkred"],
)

groups_definitions = tuple(
    {
        "classifier": AdaBoostClassifier,
        "classifier_args": {
            "n_estimators": 50,
            "algorithm": "SAMME",
            "random_state": 42,
            "learning_rate": 1,
            "estimator": DecisionTreeClassifier(max_depth=1),
        },
        "kfold_args": {
            "random_state": 42,
            "shuffle": True,
        },
        "features": {"color": param[1], "label": "train size = 0.8"},
        "rebalance": True,
    }
    for param in params
)

gomo = GOMO.GroupOfModelsOperator(X, y, groups_definitions, True, 5)

gomo.fit_all_models()
gomo.predict_targets_for_all_models_adaboost()
gomo.calculate_metrics_for_all_models()
gomo.calculate_average_metrics_for_all_groups()

GOMO.save_to_file(gomo, "by_n_estim_disease.pkl")

############################################
# maximal depth

params = (
    [1, "darkred"],
    [2, "darkblue"],
    [3, "darkgreen"],
    [4, "gold"],
)

groups_definitions = tuple(
    {
        "classifier": AdaBoostClassifier,
        "classifier_args": {
            "n_estimators": N_ESTIM,
            "algorithm": "SAMME",
            "random_state": 42,
            "learning_rate": 1,
            "estimator": DecisionTreeClassifier(max_depth=param[0]),
        },
        "kfold_args": {
            "random_state": 42,
            "shuffle": True,
        },
        "features": {"color": param[1], "label": f"max depth = {param[0]}"},
        "rebalance": True,
    }
    for param in params
)

gomo = GOMO.GroupOfModelsOperator(X, y, groups_definitions, True, 5)

gomo.fit_all_models()
gomo.predict_targets_for_all_models_adaboost()
gomo.calculate_metrics_for_all_models()
gomo.calculate_average_metrics_for_all_groups()

GOMO.save_to_file(gomo, "by_max_depth_disease.pkl")

############################################
# learn rate

params = (
    [0.05, "darkred"],
    [0.1, "darkblue"],
    [0.3, "darkgreen"],
    [0.5, "magenta"],
    [1, "gold"]
)

groups_definitions = tuple(
    {
        "classifier": AdaBoostClassifier,
        "classifier_args": {
            "n_estimators": N_ESTIM,
            "algorithm": "SAMME",
            "random_state": 42,
            "learning_rate": param[0],
            "estimator": DecisionTreeClassifier(max_depth=1),
        },
        "kfold_args": {
            "random_state": 42,
            "shuffle": True,
        },
        "features": {"color": param[1], "label": f"learn rate = {param[0]}"},
        "rebalance": True,
    }
    for param in params
)

gomo = GOMO.GroupOfModelsOperator(X, y, groups_definitions, True, 5)

gomo.fit_all_models()
gomo.predict_targets_for_all_models_adaboost()
gomo.calculate_metrics_for_all_models()
gomo.calculate_average_metrics_for_all_groups()

GOMO.save_to_file(gomo, "by_learn_rate_disease.pkl")
