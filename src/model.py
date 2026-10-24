import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# 1. Load the cached clean data
df = pd.read_csv('data/clean_merged_dataset.csv')

# 2. Define a function to extract the sensor readings from the JSON string
def parse_sensor_json(json_string):
    """
    Parses a raw JSON string of sensor payloads into a dictionary of environmental features.

    Args:
        json_string (str): A stringified JSON payload containing sensor variable names and values.

    Returns:
        dict: A dictionary mapping sensor names (e.g., 'Temperature') to their numeric values.
              Returns an empty dictionary if the JSON parsing fails.
    """
    try:
        data = json.loads(json_string)
        features = {}
        for item in data:
            name = item['variable']['name']
            value = item['values'][0]['value']
            features[name] = value
        return features
    except:
        return {}

print("Unpacking JSON features...")

# 3. Apply the function and expand the dictionary into individual columns
extracted_features = df['jsondata'].apply(parse_sensor_json).apply(pd.Series)

# Combine the new numeric columns back with the original dataframe
df = pd.concat([df, extracted_features], axis=1)

# 4. Extract temporal features
df['time'] = pd.to_datetime(df['time'])
df['Hour'] = df['time'].dt.hour
df['DayOfWeek'] = df['time'].dt.day_of_week  

# Convert to Binary Classification: Occupied (1) vs Empty (0)
# CO2 cannot physically distinguish between "Current" and "Recent" due to slow gas decay
df['is_occupied'] = df['occupancystatus'].apply(lambda x: 0 if x == 'NotOccupied' else 1)

# Define features (X) and target (y)
X = df[['Temperature', 'Humidity', 'Carbon dioxide', 'Pressure', 
        'Light Volatile Organic Compounds', 'Hour', 'DayOfWeek']]
y = df['is_occupied']

clean_idx = X.dropna().index
X = X.loc[clean_idx]
y = y.loc[clean_idx]

# 5. Random split (shuffle=True) to eliminate seasonal drift
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 6. Run the Binary Dummy Classifier baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_predictions = dummy.predict(X_test)

print("\n--- Binary Baseline (Dummy) Results ---")
print(f"Accuracy: {accuracy_score(y_test, dummy_predictions):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, dummy_predictions, zero_division=0))

# 7. Train the Binary Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\n--- Final Binary Random Forest Results ---")
print(f"Accuracy: {rf_accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions, zero_division=0))