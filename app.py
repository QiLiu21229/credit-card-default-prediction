
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from tensorflow import keras

# Load the trained neural network model.
model = keras.models.load_model("tuned_model.keras")

# Load the fitted data preprocessor.
preprocessor = joblib.load("preprocessor.joblib")

# Define categorical features used during training.
category_columns = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

def prepare_categories(data):
    # Create a copy of the input data.
    result = data.copy()

    # Convert categorical values to strings.
    for column in category_columns:
        result[column] = result[column].astype(str)

    # Apply the same rules used during training.
    result["EDUCATION"] = result["EDUCATION"].replace({
        "0": "unknown",
        "5": "unknown",
        "6": "unknown"
    })

    result["MARRIAGE"] = result["MARRIAGE"].replace({
        "0": "unknown"
    })

    return result

# Define the prediction function.
def predict_default(customer_data):
    # Convert the input dictionary into a DataFrame.
    input_data = pd.DataFrame([customer_data])

    # Apply the same categorical processing used in training.
    prepared_data = prepare_categories(input_data)

    # Apply the saved preprocessing transformer.
    processed_data = preprocessor.transform(prepared_data)

    # Convert the processed data to float32.
    if hasattr(processed_data, "toarray"):
        processed_data = processed_data.toarray()

    processed_data = np.asarray(
        processed_data, dtype=np.float32
    )

    # Predict the probability of credit card default.
    probability = float(
        model.predict(processed_data, verbose=0)[0][0]
    )

    # Apply the classification threshold of 0.50.
    prediction = 1 if probability >= 0.50 else 0

    return prediction, probability



# Define the original input features.
feature_names = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# Create a function to connect the inputs to the model.
def run_prediction(*values):
    customer_data = dict(zip(feature_names, values))

    if any(value is None for value in values):
        return "Please complete all input fields.", None

    prediction, probability = predict_default(customer_data)

    if prediction == 1:
        result = "Predicted Default"
    else:
        result = "Predicted No Default"

    return result, round(probability, 4)


# Build the Gradio web interface.
with gr.Blocks(title="Credit Card Default Prediction") as demo:

    gr.Markdown("# Credit Card Default Prediction")
    gr.Markdown(
        "Predict credit card default risk using a trained neural network."
    )

    with gr.Accordion("Customer Information", open=True):
        with gr.Row():
            limit_bal = gr.Number(label="Credit Limit", value=50000)
            age = gr.Number(label="Age", value=30, precision=0)

        with gr.Row():
            sex = gr.Number(label="Sex (1=Male, 2=Female)",
                            value=1, precision=0)
            education = gr.Number(label="Education Code",
                                   value=2, precision=0)
            marriage = gr.Number(label="Marriage Code",
                                  value=1, precision=0)

    with gr.Accordion("Repayment History", open=False):
        gr.Markdown(
            "Repayment status codes: -2, -1, 0, 1, 2, etc."
        )

        pay_inputs = [
            gr.Number(label=name, value=0, precision=0)
            for name in [
                "PAY_0", "PAY_2", "PAY_3",
                "PAY_4", "PAY_5", "PAY_6"
            ]
        ]

    with gr.Accordion("Billing and Payment Amounts", open=False):
        bill_inputs = [
            gr.Number(label=f"BILL_AMT{i}", value=0)
            for i in range(1, 7)
        ]

        payment_inputs = [
            gr.Number(label=f"PAY_AMT{i}", value=0)
            for i in range(1, 7)
        ]

    predict_button = gr.Button(
        "Predict Default Risk",
        variant="primary"
    )

    gr.Markdown("## Prediction Result")

    prediction_output = gr.Textbox(
        label="Predicted Class"
    )

    probability_output = gr.Number(
        label="Default Probability (0-1)"
    )

    # Match the original training feature order.
    inputs = [
        limit_bal, sex, education, marriage, age,
        *pay_inputs,
        *bill_inputs,
        *payment_inputs
    ]

    predict_button.click(
        fn=run_prediction,
        inputs=inputs,
        outputs=[prediction_output, probability_output]
    )


# Launch the web application.
if __name__ == "__main__":
    demo.launch()
