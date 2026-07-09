import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("IMDb Movies India.csv",encoding="latin1")
print(df.head())
print(df.info())
print(df.describe(include="all"))

df = df.dropna().copy()

df["Year"] = df["Year"].str.extract(r"(\d{4})").astype(int)
df["Duration"]  = df["Duration"].str.extract(r"(\d+)").astype(int)
df["Votes"]=df["Votes"].str.replace(",","",regex=False).astype(int)
print(df[["Year", "Duration", "Votes"]].head())

num_cols = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10, 8))
sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Genre bar chart (before encoding)
top_genres = (
    df["Genre"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10, 5))
top_genres.plot(kind="bar")
plt.title("Top 10 Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

le = LabelEncoder()
for col in ["Genre","Director","Actor 1","Actor 2","Actor 3"]:
    df[col]=le.fit_transform(df[col])

X = df[["Year", "Duration", "Votes",
        "Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]]

y = df["Rating"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred[:5])

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("R2 score:",r2)

print(pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
}))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Ratings")
plt.show()

