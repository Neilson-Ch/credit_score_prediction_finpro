            "Annual Income",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
        )
        monthly_salary = st.number_input(
            "Monthly Inhand Salary",
            min_value=0.0,
            value=4000.0,
            step=100.0,
        )
        num_bank_accounts = st.number_input(
            "Num Bank Accounts",
            min_value=1,
            max_value=20,
            value=3,
        )
        num_credit_card = st.number_input(
            "Num Credit Card",
            min_value=0,
            max_value=30,
            value=4,
        )
        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            value=12.0,
        )
        num_of_loan = st.number_input("Num of Loan", min_value=0, value=2)
        loan_types = st.multiselect(
            "Type of Loan",
            LOAN_OPTIONS,
            default=["Auto Loan"],
        )
        delay_due = st.number_input("Delay from Due Date (hari)", value=5)
        num_delayed_payment = st.number_input(
            "Num of Delayed Payment",
            min_value=0,
            value=3,
        )

    with col2:
        changed_credit_limit = st.number_input(
            "Changed Credit Limit",
            value=5.5,
        )
        num_credit_inquiries = st.number_input(
            "Num Credit Inquiries",
            min_value=0.0,
            value=2.0,
        )
        credit_mix = st.selectbox(
            "Credit Mix",
            ["Good", "Standard", "Bad", "Unknown"],
        )
        outstanding_debt = st.number_input(
            "Outstanding Debt",
            min_value=0.0,
            value=500.0,
        )
        credit_util = st.number_input(
            "Credit Utilization Ratio (%)",
            min_value=0.0,
            value=30.0,
        )
        history_years = st.number_input(
            "Credit History (tahun)",
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
            "Payment of Min Amount",
            ["Yes", "No", "NM"],
        )
        total_emi = st.number_input(
            "Total EMI per Month",
            min_value=0.0,
            value=100.0,
        )
        amount_invested = st.number_input(
            "Amount Invested Monthly",
            min_value=0.0,
            value=50.0,
        )
        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "Low_spent_Small_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
            ],
        )
        monthly_balance = st.number_input("Monthly Balance", value=300.0)

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
        label = response["labels"][0]
        probabilities = response["probabilities"][0]
        probability_map = {
            LABEL_MAP[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }
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
