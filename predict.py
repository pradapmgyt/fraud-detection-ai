import joblib
import numpy as np

# Load model
model = joblib.load('models/fraud_model.pkl')

# Example input (replace with real later)
# 30 features required (same as dataset)
sample_input = np.random.rand(1, 30)

# Predict
prediction = model.predict(sample_input)

# Output
if prediction[0] == 1:
    print("🚨 Fraud Transaction Detected!")
else:
    print("✅ Safe Transaction")