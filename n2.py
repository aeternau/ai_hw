import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, mean_absolute_error

df = pd.read_csv("school_progress.csv")
features = ["grade", "attempts", "hours_spent"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["student_id"]))

train = df.iloc[train_idx]
test = df.iloc[test_idx]

if train["student_id"].isin(test["student_id"]).sum() == 0:
    print("общих нет")
else:
    print("общие есть")

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(train[features], train["status"])

s = pd.DataFrame(
    {"fact": test["status"], "predict": model.predict(test[features])}
)
print(s.head(10))
