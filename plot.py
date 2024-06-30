import matplotlib.pyplot as plt

import GOMO
from GOMO import *


def plot_acc_prec_rec(gomo, name):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"Accuracy {name}")
    ax.set_xlabel("number of estimators")
    ax.legend()
    for group in gomo.groups:
        for model in group.models:
            ax.plot(model.accuracy, color=group.features["color"], alpha=0.15)
        ax.plot(
            group.average_accuracy,
            color=group.features["color"],
            label=group.features["label"],
        )
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"Precision {name}")
    ax.set_xlabel("number of estimators")
    ax.legend()
    for group in gomo.groups:
        for model in group.models:
            ax.plot(model.precision, color=group.features["color"], alpha=0.15)
        ax.plot(
            group.average_precision,
            color=group.features["color"],
            label=group.features["label"],
        )
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"Recall {name}")
    ax.set_xlabel("number of estimators")
    ax.legend()
    for group in gomo.groups:
        for model in group.models:
            ax.plot(model.recall, color=group.features["color"], alpha=0.15)
        ax.plot(
            group.average_recall,
            color=group.features["color"],
            label=group.features["label"],
        )
    ax.legend()

plot_acc_prec_rec(GOMO.load_from_file("./by_n_estim_insurance.pkl"), "ins_n_estim")
plot_acc_prec_rec(GOMO.load_from_file("./by_max_depth_insurance.pkl"), "ins_max_depth")
plot_acc_prec_rec(GOMO.load_from_file("./by_learn_rate_insurance.pkl"), "ins_learn_rate")

plot_acc_prec_rec(GOMO.load_from_file("./by_n_estim_disease.pkl"), "dis_n_estim")
plot_acc_prec_rec(GOMO.load_from_file("./by_max_depth_disease.pkl"), "dis_max_depth")
plot_acc_prec_rec(GOMO.load_from_file("./by_learn_rate_disease.pkl"), "dis_learn_rate")

plt.show()
