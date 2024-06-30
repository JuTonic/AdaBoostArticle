# AdaBoostArticle

install poetry https://python-poetry.org/, git and 7zip https://7-zip.org/:

```
winget install 7zip git

winget install poetry
# or
pip isntall poetry
```

clone the repo:

```
git clone https://github.com/JuTonic/AdaBoostArticle.git
```

Run poetry in the repo folder:
```
poetry update
```

Wait until it install all dependencies.

To train models on insurance dataset and save results run:
```
poetry run python insurance.py
```

For heart disease dataset:
```
poetry run python heartdisease.py
```

To generate and show plot on these models:
```
poetry run python plot.py
```

To generate plot for computational complexity part run:
```
poetry run python computation_time_and_tuning.py
```

extract computes.7z

