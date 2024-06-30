import time

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import GOMO

# Load and categorize insurance dataset
df = pd.read_csv("./data/insurance_data.csv")

y = df["claim_status"]
X = df.drop(["claim_status", "policy_id"], axis=1)

le = LabelEncoder()

# Variables to categorize
requires_transform = [
    "region_code",
    "segment",
    "model",
    "fuel_type",
    "engine_type",
    "is_esc",
    "is_adjustable_steering",
    "is_tpms",
    "is_parking_sensors",
    "is_parking_camera",
    "rear_brakes_type",
    "transmission_type",
    "steering_type",
    "is_front_fog_lights",
    "is_rear_window_wiper",
    "is_rear_window_washer",
    "is_rear_window_defogger",
    "is_brake_assist",
    "is_power_door_locks",
    "is_central_locking",
    "is_power_steering",
    "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror",
    "is_ecw",
    "is_speed_alert",
]

for i in requires_transform:
    X[i] = le.fit_transform(X[i])

# dropping variables not suitable for model fitting
X = X.drop(["max_power", "max_torque"], axis=1)


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
            "n_estimators": N_ESTIM,
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

GOMO.save_to_file(gomo, "by_n_estim_insurance.pkl")

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

GOMO.save_to_file(gomo, "by_max_depth_insurance.pkl")

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

GOMO.save_to_file(gomo, "by_learn_rate_insurance.pkl")
