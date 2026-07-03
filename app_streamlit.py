"""Streamlit client untuk Credit Score endpoint di Amazon SageMaker.

Seluruh preprocessing dan loading artifact dilakukan oleh inference.py di endpoint.
Aplikasi EC2 hanya membuat payload mentah, memanggil endpoint, dan menampilkan hasil.
"""

import json
import os

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="centered",
)

st.title("💳 Credit Score Prediction")
st.caption("Prediksi kategori credit score nasabah: Poor / Standard / Good")

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get(
    "AWS_REGION",
    os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)

LABEL_MAP = {0: "Poor", 1: "Standard", 2: "Good"}


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(raw_record: dict) -> dict:
    response = get_runtime_client().invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps({"instances": [raw_record]}),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def parse_prediction_response(response) -> tuple[str, dict[str, float]]:
    """Normalisasi dua format response inference yang pernah digunakan."""
    if isinstance(response, list):
        if not response:
            raise ValueError("Endpoint mengembalikan list kosong")
        response = response[0]

    if not isinstance(response, dict):
        raise ValueError(
            f"Format response endpoint tidak dikenali: {type(response).__name__}"
        )

    # Contract inference lama:
    # {"prediction": "Standard", "probability": {"Poor": ..., ...}}
    if "prediction" in response and "probability" in response:
        label = str(response["prediction"])
        probability_map = {
            str(name): float(value)
            for name, value in response["probability"].items()
        }
        return label, probability_map

    # Contract inference baru:
    # {"labels": ["Standard"], "probabilities": [[...]]}
    if "labels" in response and "probabilities" in response:
        label = str(response["labels"][0])
        probabilities = response["probabilities"][0]
        probability_map = {
            LABEL_MAP[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }
        return label, probability_map

    raise ValueError(
        f"Key response endpoint tidak dikenali: {list(response.keys())}"
    )


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
]

LOAN_OPTIONS = [
    "Auto Loan",
    "Credit-Builder Loan",
    "Personal Loan",
    "Home Equity Loan",
    "Mortgage Loan",
    "Student Loan",
    "Debt Consolidation Loan",
    "Payday Loan",
]

OCCUPATION_OPTIONS = [
    "Engineer",
    "Lawyer",
    "Architect",
    "Media_Manager",
    "Accountant",
    "Entrepreneur",
    "Developer",
    "Scientist",
    "Teacher",
    "Mechanic",
    "Journalist",
    "Doctor",
    "Manager",
    "Musician",
    "Writer",
    "Unknown",
]


with st.form("credit_form"):
    st.subheader("Data Nasabah")

    col1, col2 = st.columns(2)

    with col1:
        month = st.selectbox("Month", MONTHS)
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
        )
        monthly_salary = st.number_input(
            "Monthly_Inhand_Salary",
            min_value=0.0,
            value=4000.0,
            step=100.0,
        )
        num_bank_accounts = st.number_input(
            "Num_Bank_Accounts",
            min_value=1,
            max_value=20,
            value=3,
        )
        num_credit_card = st.number_input(
            "Num_Credit_Card",
            min_value=0,
            max_value=30,
            value=4,
        )
        interest_rate = st.number_input(
            "Interest_Rate (%)",
            min_value=0.0,
            value=12.0,
        )
        num_of_loan = st.number_input("Num of Loan", min_value=0, value=2)
        loan_types = st.multiselect(
            "Type_of_Loan",
            LOAN_OPTIONS,
            default=["Auto Loan"],
        )
        delay_due = st.number_input("Delay from Due Date (hari)", value=5)
        num_delayed_payment = st.number_input(
            "Num_of_Delayed_Payment",
            min_value=0,
            value=3,
        )

    with col2:
        changed_credit_limit = st.number_input(
            "Changed_Credit_Limit",
            value=5.5,
        )
        num_credit_inquiries = st.number_input(
            "Num_Credit_Inquiries",
            min_value=0.0,
            value=2.0,
        )
        credit_mix = st.selectbox(
            "Credit_Mix",
            ["Good", "Standard", "Bad", "Unknown"],
        )
        outstanding_debt = st.number_input(
            "Outstanding_Debt",
            min_value=0.0,
            value=500.0,
        )
        credit_util = st.number_input(
            "Credit_Utilization_Ratio",
            min_value=0.0,
            value=30.0,
        )
        history_years = st.number_input(
            "Credit_History (tahun)",
            min_value=0,
            value=10,
        )
        history_months = st.number_input(
            "Credit History (bulan tambahan)",
            min_value=0,
            max_value=11,
            value=3,
        )
        payment_min = st.selectbox(
            "Payment_of_Min_Amount",
            ["Yes", "No", "NM"],
        )
        total_emi = st.number_input(
            "Total_EMI_per_Month",
            min_value=0.0,
            value=100.0,
        )
        amount_invested = st.number_input(
            "Amount_Invested_Monthly",
            min_value=0.0,
            value=50.0,
        )
        payment_behaviour = st.selectbox(
            "Payment_Behaviour",
            [
                "Low_spent_Small_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
            ],
        )
        monthly_balance = st.number_input("Monthly_Balance", value=300.0)

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)


if submitted:
    type_of_loan = ", and ".join(loan_types) if loan_types else "Not Specified"

    # Payload sengaja tetap mentah. Cleaning dan MultiLabelBinarizer dijalankan
    # oleh inference.py di SageMaker endpoint.
    raw_input = {
        "Month": month,
        "Age": int(age),
        "Occupation": occupation,
        "Annual_Income": str(annual_income),
        "Monthly_Inhand_Salary": float(monthly_salary),
        "Num_Bank_Accounts": int(num_bank_accounts),
        "Num_Credit_Card": int(num_credit_card),
        "Interest_Rate": float(interest_rate),
        "Num_of_Loan": str(num_of_loan),
        "Type_of_Loan": type_of_loan,
        "Delay_from_due_date": int(delay_due),
        "Num_of_Delayed_Payment": str(num_delayed_payment),
        "Changed_Credit_Limit": str(changed_credit_limit),
        "Num_Credit_Inquiries": float(num_credit_inquiries),
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": str(outstanding_debt),
        "Credit_Utilization_Ratio": float(credit_util),
        "Credit_History_Age": (
            f"{history_years} Years and {history_months} Months"
        ),
        "Payment_of_Min_Amount": payment_min,
        "Total_EMI_per_month": float(total_emi),
        "Amount_invested_monthly": str(amount_invested),
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": str(monthly_balance),
    }

    try:
        response = invoke_endpoint(raw_input)
        label, probability_map = parse_prediction_response(response)
    except NoCredentialsError:
        st.error(
            "AWS credentials tidak ditemukan. Attach IAM instance profile "
            "yang memiliki izin sagemaker:InvokeEndpoint."
        )
    except ClientError as error:
        message = error.response.get("Error", {}).get("Message", str(error))
        st.error(f"AWS error: {message}")
    except Exception as error:
        st.error(f"Gagal melakukan prediksi: {error}")
    else:
        st.divider()
        st.subheader("Hasil Prediksi")

        icon = {"Poor": "🔴", "Standard": "🟡", "Good": "🟢"}.get(
            label,
            "⚪",
        )
        st.markdown(f"### {icon} **{label}**")

        probability_frame = pd.DataFrame(
            {
                "Kelas": list(probability_map.keys()),
                "Probabilitas": list(probability_map.values()),
            }
        ).sort_values("Probabilitas", ascending=False)

        st.bar_chart(probability_frame.set_index("Kelas"))
        st.dataframe(
            probability_frame,
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("Lihat raw JSON response"):
            st.json(response)
