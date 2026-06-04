import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sys

# 1. CSV file importation
try:
    data = pd.read_csv(
        "results.csv",
        sep=',',
        decimal='.',
        header=None,  # Tell Python: "The file does not have a header line"
        names=['Run Order', 'Distance', 'Flow-rate', 'Voltage', 'Picture', 'Score Global', 'Classe']
        # Name columns in the exact order of the exported CSV file
    )

    # Security check: If the previous script included text headers,
    # detect the header row and remove it to keep only numerical data.
    if str(data.iloc[0]['Run Order']).strip() == 'Run Order':
        data = data.iloc[1:].reset_index(drop=True)

except Exception as e:
    print(f"[ERROR] Unable to load the CSV file: {e}")
    sys.exit(1)

print("--- DIAGNOSTIC: Your structured dataset ---")
print(data)
print("-" * 55 + "\n")

# 2. Data cleaning and type conversion (handling potential Excel comma issues)
for col in ['Distance', 'Flow-rate', 'Voltage', 'Classe']:
    data[col] = data[col].astype(str).str.replace(',', '.')
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 3. Remove rows with missing essential values
data = data.dropna(subset=['Distance', 'Flow-rate', 'Voltage', 'Classe'])

if data.empty:
    print("[CRITICAL ERROR] The dataset is empty after cleaning. Please check your raw data.")
    sys.exit(1)

# 4. Splitting features (X) and the target class to predict (y)
X = data[['Distance', 'Flow-rate', 'Voltage']]
y = data['Classe']

# 5. KNN model initialization and training (k=1 because of single reference tracking)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# 6. Prediction test on a new machine setting
# Modify these values to test your predictions: [[Distance, Flow-rate, Voltage]]
new_setting = pd.DataFrame([[16, 0.7, 15]], columns=['Distance', 'Flow-rate', 'Voltage'])
prediction = knn.predict(new_setting)

print("--- PREDICTION RESULT ---")
print(f"For these settings, the KNN predicts class: {int(prediction[0])}")
print("-" * 33)