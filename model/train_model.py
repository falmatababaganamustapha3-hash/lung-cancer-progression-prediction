import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# 1. Load the cleaned dataset
df = pd.read_csv('../data/lung_cancer_data_cleaned.csv')

# 2. Select features (drop identifiers and the text label, keep only numeric predictors)
X = df.drop(columns=['Patient_Id', 'Level', 'Level_Score'])
y = df['Level_Score']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 6. Save the trained model + the feature column order (needed later by Flask)
joblib.dump(model, 'lung_cancer_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')

print("\nModel saved as model/lung_cancer_model.pkl")