from matplotlib.colors import LightSource
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import time
import pickle
from scipy.interpolate import griddata
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from labellines import labelLine, labelLines
from matplotlib.pyplot import get_cmap
import GOMO

##### Uncomment to compute again and save results to file (WILL TAKE SOME TIME)

# df = pd.read_csv("./heart_disease.csv")
#
# y = df["HeartDiseaseorAttack"]
# X = df.drop(["HeartDiseaseorAttack"], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state=42)
#
# smote = SMOTE(random_state=42)
#
# X_train, y_train = smote.fit_resample(X_train, y_train)
#
# comp_time = []
#
# def estimate_time(n, max_depth):
#     print(f"Fitting model n = {n}, max_depth = {max_depth}")
#     abc = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n, algorithm="SAMME")
#     start = time.time()
#     model = abc.fit(X_train, y_train)
#     took1 = time.time() - start
#
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#
#     start = time.time()
#     model = abc.fit(X_train, y_train)
#     took2 = time.time() - start
#     return (n, max_depth, (took1 + took2) / 2, acc, prec, rec)
#
# comp_time = []
#
# with ThreadPoolExecutor(max_workers=6) as executor:
#     futures = []
#     for n in [5*(n + 1) for n in range(16)]:
#         for max_depth in [i + 1 for i in range(16)]:
#             futures.append(executor.submit(estimate_time, n, max_depth))
#
#     for future in as_completed(futures):
#         comp_time.append(future.result())
#
# def save_to_file(obj, filename: str = "obj.pkl"):
#     with open(filename, "wb") as f:
#         pickle.dump(obj, f)
#
# save_to_file(comp_time, "comp_time.pkl")

data = GOMO.load_from_file("comp_time.pkl")
data = np.array(data)

fig = plt.figure()
ax = fig.add_subplot()

ndc = {"n": data[:, 0], "d": data[:, 1], "c": data[:, 2], "acc": data[:, 3], "prec": data[:, 4], "rec": data[:, 5]}

df = pd.DataFrame(ndc)

print(df)

ni = np.linspace(10, 80, 16)
di = np.linspace(1, 16, 16)


cmap = get_cmap('RdYlBu')
# cmap = get_cmap('viridis')

for i in range(8):
    i = 2 * i
    x = ni
    y = np.array(df[df['d'] == i + 1]['c'])
    ax.plot(x, y, color = cmap(i / 15), label = f"max depth = {i + 1}")
    # ax.text(x[-1], y[-1], f"{i + 1}", color = cmap(i / 15))

labelLines(ax.get_lines(), fontsize=10, zorder=2.5)
ax.set_xlabel("number of estimators")
ax.set_ylabel("computation time (seconds)")
fig.savefig("noo-ct.pdf")

fig = plt.figure()
ax = fig.add_subplot()

for i in range(8):
    i = 2 * i
    x = di
    y = np.array(df[df['n'] == (i + 1) * 5]['c'])
    ax.plot(x, y, color = cmap(i / 15), label = f"num of obs = {(i + 1) * 5}")

from labellines import labelLine, labelLines
labelLines(ax.get_lines(), align=True, fontsize=10, zorder=2.5)
ax.set_xlabel("max depth")
ax.set_ylabel("computation time (seconds)")
fig.savefig("md-ct.pdf")

ni, di = np.meshgrid(ni, di)

ci = griddata((ndc['n'], ndc['d']), ndc['c'], (ni, di), method='nearest')


def plotcontour(z):
    fig = plt.figure()
    ax = fig.add_subplot()
    CS = ax.contour(ni, di, z, cmap="RdYlBu")
    ax.clabel(CS)
    ax.set_xlabel("number of estimators")
    ax.set_ylabel("max depth")
    fig.savefig("noo-md.pdf")

plotcontour(ci)

def plot3d(x, y, z, z_min_max, xlabel, ylabel, zlabel, filepath, x_offset = 0, y_offset = 17, plot_surface_args = {"cmap": "viridis"}):
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    ax.plot_surface(x, y, z, **plot_surface_args)
    ax.contourf(x, y, z, zdir='z', offset=z_min_max[0] * 0.995, cmap='RdYlBu')
    ax.contourf(x, y, z, zdir='x', offset=x_offset, cmap='RdYlBu')
    ax.contourf(x, y, z, zdir='y', offset=y_offset, cmap='RdYlBu')
    ax.set(
            xlim=(0, 90), 
            ylim=(0, 17), 
            zlim=(z_min_max[0]*0.99, z_min_max[1] * 1.01),
            xlabel='number of estimators', 
            ylabel='max depth', 
            zlabel=zlabel
    )
    fig.savefig(filepath)


plot3d(ni, di, ci, (ndc['c'].min(), ndc['c'].max()),"number of observation", "max depth", "computational time (seconds)", "comp_time.pdf", plot_surface_args={"alpha": 0.3, "edgecolor": "black"})

ci = griddata((ndc['n'], ndc['d']), ndc['prec'], (ni, di), method='nearest')

plot3d(ni, di, ci, (ndc['prec'].min(), ndc['prec'].max()),"number of observation", "max depth", "precision", "prec_3d.pdf")

ci = griddata((ndc['n'], ndc['d']), ndc['rec'], (ni, di), method='nearest')

plot3d(ni, di, ci, (ndc['rec'].min(), ndc['rec'].max()),"number of observation", "max depth", "recall", "recall_3d.pdf")

ci = griddata((ndc['n'], ndc['d']), ndc['acc'], (ni, di), method='nearest')

plot3d(ni, di, ci, (ndc['acc'].min(), ndc['acc'].max()),"number of observation", "max depth", "accuracy", "accuracy_3d.pdf")

plt.show()
