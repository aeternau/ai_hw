import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("school_progress.csv")

X = df[["grade", "attempts", "hours_spent"]]
y = df["status"]

print(X.shape)
print(y.shape)

print("Распределение классов status:")
print(y.value_counts())

# модель будет определять статус ученика по оценкам,
# попыткам и потраченным часам. это не прогноз т.к.
# мы определяем уже случившийся результат по итогам
