
import json
import joblib
from tensorflow import keras

# Load the trained model and preprocessor.
model = keras.models.load_model(
    "tuned_model.keras"
)
preprocessor = joblib.load(
    "preprocessor.joblib"
)

# Extract categorical encoding information.
encoder = preprocessor.named_transformers_["categories"]
scaler = preprocessor.named_transformers_["numbers"]

category_columns = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

numeric_columns = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# Export trained neural network weights.
layers_data = []

for layer in model.layers:
    weights = layer.get_weights()

    if weights:
        layers_data.append({
            "weights": weights[0].tolist(),
            "biases": weights[1].tolist(),
            "activation": layer.activation.__name__
        })

# Export model and preprocessing parameters.
export_data = {
    "category_columns": category_columns,
    "numeric_columns": numeric_columns,
    "categories": [
        values.tolist()
        for values in encoder.categories_
    ],
    "means": scaler.mean_.tolist(),
    "scales": scaler.scale_.tolist(),
    "layers": layers_data,
    "threshold": 0.5
}

# Save the exported data.
with open("model_data.json", "w") as file:
    json.dump(export_data, file, indent=2)

print("Model exported successfully.")
