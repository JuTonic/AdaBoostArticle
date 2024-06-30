import lzma
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

smote = SMOTE(random_state=42)

class Model(object):
    def __init__(
        self,
        classifier,
        classifier_args: dict,
        train_index,
        test_index,
        rebalance: bool = True,
    ):
        """
        abc_args is a dictionary with arguments for AdaBoostClassifier from sklearn,
        """
        self.abc = classifier(**classifier_args)

        self.train_index, self.test_index = train_index, test_index

        self.rebalance = rebalance


    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X_train, y_train = X.iloc[self.train_index], y.iloc[self.train_index]

        if self.rebalance:
            while True:
                try:
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    break
                except:
                    print("One error")
                    continue

        try:
            start = time.time()
            self.model = self.abc.fit(X_train, y_train)
            end = time.time()
            self.time_fitted = end - start
        except Exception as exc:
            print(exc, len(X_train), len(y_train))

    def predict_target_from_1_to_n(self, X: pd.DataFrame):  # AdaBoostSpecific
        estimator_pred = []
        
        X_test = X.iloc[self.test_index]

        range_len_test_sample = range(len(X_test))

        for estimator in self.model.estimators_:
            estimator_pred.append(estimator.predict(X_test.values))

        pos = [0 for _ in range_len_test_sample]
        neg = [0 for _ in range_len_test_sample]


        self.Y_PRED = []

        for i in range(len(self.model.estimators_)):
            y_pred = [0 for _ in range_len_test_sample]

            not_estimator_pred = -1 * (estimator_pred[i] - 1)
            pos = np.array(pos) + estimator_pred[i] * np.array(
                self.model.estimator_weights_[i]
            )
            neg = np.array(neg) + not_estimator_pred * np.array(
                self.model.estimator_weights_[i]
            )
            for j in range_len_test_sample:
                y_pred[j] = 1 if pos[j] > neg[j] else 0
            self.Y_PRED.append(y_pred)

    def calculate_metrics(self, y: pd.DataFrame):
        self.accuracy = []
        self.precision = []
        self.recall = []

        y_test = y.iloc[self.test_index]

        for y_pred in self.Y_PRED:
            self.accuracy.append(accuracy_score(y_test, y_pred))
            self.precision.append(precision_score(y_test, y_pred))
            self.recall.append(recall_score(y_test, y_pred))


class GroupOfModels(object):
    def __init__(self, splits, definition: dict):
        self.models = []
        self.average_over = len(splits)

        for train_index, test_index in splits:
            self.models.append(
                Model(
                    definition["classifier"],
                    definition["classifier_args"],
                    train_index,
                    test_index,
                    definition["rebalance"],
                )
            )

        self.features = definition["features"]

    def calculate_average_metrics(self):
        self.average_accuracy = (
            sum([np.array(model.accuracy) for model in self.models]) / self.average_over
        )
        self.average_precision = (
            sum([np.array(model.precision) for model in self.models])
            / self.average_over
        )
        self.average_recall = (
            sum([np.array(model.recall) for model in self.models]) / self.average_over
        )


class GroupOfModelsOperator(object):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, groups_definition, k_fold: bool, average_over: int):
        print(
            "initializing models",
            f"params:\n average_over = {average_over}",
        )
        self.groups = []

        if k_fold:
            try:
                kfold_args = groups_definition["kfold_args"]
            except:
                kfold_args = {"shuffle": True, "random_state": 42}
            kf = KFold(n_splits = average_over, **kfold_args)
            split = list(kf.split(X))

        else:
            for definition in groups_definition:
                for i in range(average_over):
                    try:
                        tts_args = definition["tts_args"]
                        split.append(train_test_split(range(len(X), random_state = i, **tts_args)))
                    except:
                        print("No tts args provided")
                self.groups.append(
                    GroupOfModels(split, definition)
                )
        
        for definition in groups_definition:
            self.groups.append(
                GroupOfModels(split, definition)
            )

        self.average_over = average_over
        self.X = X
        self.y = y

    def fit_all_models(self):
        print("fitting models")
        need_to_fit_models = self.average_over * len(self.groups)
        models_fitted = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for group in self.groups:
                for model in group.models:
                    futures.append(executor.submit(model.fit, self.X, self.y))

            for future in as_completed(futures):
                try:
                    models_fitted += 1
                    print(f"fitted {models_fitted} models out of {need_to_fit_models}")
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")

    def predict_targets_for_all_models_adaboost(self):
        print("calculating predicts")
        need_to_calculate = self.average_over * len(self.groups)
        calculated = 0
        with ThreadPoolExecutor(max_workers=22) as executor:
            futures = []
            for group in self.groups:
                for model in group.models:
                    futures.append(executor.submit(model.predict_target_from_1_to_n, self.X))

            for future in as_completed(futures):
                try:
                    calculated += 1
                    print(f"calculated {calculated} out of {need_to_calculate}")
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")

    def calculate_metrics_for_all_models(self):
        print("calculating metrics")
        need_to_calculate = self.average_over * len(self.groups)
        calculated = 0
        with ThreadPoolExecutor(max_workers=22) as executor:
            futures = []
            for group in self.groups:
                for model in group.models:
                    futures.append(executor.submit(model.calculate_metrics, self.y))

            for future in as_completed(futures):
                try:
                    calculated += 1
                    print(f"calculated {calculated} out of {need_to_calculate}")
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")

    def calculate_average_metrics_for_all_groups(self):
        print("calculating averages")
        for group in self.groups:
            group.calculate_average_metrics()


def save_to_file(obj, filename: str = "obj.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_from_file(filename: str = "obj.pkl"):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj

