import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load CSV
df = pd.read_csv("playground_data.csv")

# Encode categorical variables
le_weekday = LabelEncoder()
le_weather = LabelEncoder()
le_crowd = LabelEncoder()

df["weekday"] = le_weekday.fit_transform(df["weekday"])
df["weather"] = le_weather.fit_transform(df["weather"])
df["crowd"] = le_crowd.fit_transform(df["crowd"])

X = df[["playground_id", "hour", "weekday", "weather"]]
y = df["crowd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump({
        "le_weekday": le_weekday,
        "le_weather": le_weather,
        "le_crowd": le_crowd
    }, f)

print("✅ Model and encoders saved!")

