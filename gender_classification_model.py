import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==============================
# 1. Load datasets
# ==============================
flights = pd.read_csv("data/flights.csv")
hotels = pd.read_csv("data/hotels.csv")
users = pd.read_csv("data/users.csv")

# ==============================
# 2. Merge datasets
# ==============================
flight_data = pd.merge(flights, users, left_on="userCode", right_on="code")
hotel_data = pd.merge(hotels, users, left_on="userCode", right_on="code")

# ==============================
# 3. Feature Engineering
# ==============================

# Flight behavior features
flight_features = flight_data.groupby("userCode").agg({
    "price": "mean",
    "distance": "mean",
    "time": "mean"
}).reset_index()

# Hotel behavior features
hotel_features = hotel_data.groupby("userCode").agg({
    "total": "mean",
    "days": "mean"
}).reset_index()

# Merge flight + hotel features
user_behavior = pd.merge(flight_features, hotel_features, on="userCode", how="outer")

# Add user info (gender, age)
user_behavior = pd.merge(user_behavior, users, left_on="userCode", right_on="code")

# Fill missing values
user_behavior = user_behavior.fillna(0)

# ==============================
# 4. Preprocessing
# ==============================
df = user_behavior.copy()

# Encode gender (target variable)
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# Save encoder (IMPORTANT)
joblib.dump(le, "models/gender_encoder.pkl")

# Drop unnecessary columns
df = df.drop(['userCode', 'code', 'name', 'company'], axis=1)

# ==============================
# 5. Features & Target
# ==============================
X = df.drop('gender', axis=1)
y = df['gender']

# ==============================
# 6. Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 7. Train Model
# ==============================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ==============================
# 8. Evaluate Model
# ==============================
y_pred = model.predict(X_test)

# Convert predictions back to labels (Male/Female)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test_labels, y_pred_labels))

# ==============================
# 9. Save Model
# ==============================
joblib.dump(model, "models/gender_model.pkl")

print("\n✅ Gender model saved successfully!")
print("✅ Encoder saved successfully!")