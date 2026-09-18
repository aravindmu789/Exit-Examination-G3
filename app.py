import joblib
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

# Load the complete saved pipeline directly
model = joblib.load(
    "best_linear_regression_revenue_model.pkl"
)

# Get the exact feature columns used during training
feature_columns = model.feature_names_in_.tolist()


def convert_to_float(value):
    """Convert an empty form value to None."""
    if value is None or str(value).strip() == "":
        return None

    return float(value)


def prepare_input(form_data):
    """Prepare form data in the same format used during training."""

    raw_input = {
        "enterprise_group_jurisdiction": form_data.get(
            "enterprise_group_jurisdiction"
        ),

        "company_type": form_data.get(
            "company_type"
        ),

        "employees": convert_to_float(
            form_data.get("employees")
        ),

        "employees_year": convert_to_float(
            form_data.get("employees_year")
        ),

        "revenue_currency": form_data.get(
            "revenue_currency"
        ),

        "revenue_year": convert_to_float(
            form_data.get("revenue_year")
        ),

        "net_income": convert_to_float(
            form_data.get("net_income")
        ),

        "net_income_currency": form_data.get(
            "net_income_currency"
        ),

        "net_income_year": convert_to_float(
            form_data.get("net_income_year")
        ),

        "market_cap": convert_to_float(
            form_data.get("market_cap")
        ),

        "market_cap_currency": form_data.get(
            "market_cap_currency"
        ),

        "market_cap_year": convert_to_float(
            form_data.get("market_cap_year")
        ),

        "legal_units_count": convert_to_float(
            form_data.get("legal_units_count")
        ),

        "direct_subsidiaries_count": convert_to_float(
            form_data.get("direct_subsidiaries_count")
        ),

        "max_hierarchy_depth": convert_to_float(
            form_data.get("max_hierarchy_depth")
        ),

        "is_standalone_entity": int(
            form_data.get("is_standalone_entity", "0")
        )
    }

    input_df = pd.DataFrame([raw_input])

    categorical_columns = [
        "enterprise_group_jurisdiction",
        "company_type",
        "revenue_currency",
        "net_income_currency",
        "market_cap_currency"
    ]

    # Apply the same one-hot encoding used during model training
    input_df = pd.get_dummies(
        input_df,
        columns=categorical_columns,
        dtype=float
    )

    # Match the exact training columns and their order
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return input_df


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        input_df = prepare_input(request.form)

        # The pipeline performs imputation, scaling, and prediction
        prediction = model.predict(input_df)[0]

        return render_template(
            "result.html",
            prediction=prediction
        )

    except ValueError as error:
        return render_template(
            "predict.html",
            error=f"Please enter valid numeric values. Error: {error}"
        )

    except Exception as error:
        return render_template(
            "predict.html",
            error=f"Prediction error: {error}"
        )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )