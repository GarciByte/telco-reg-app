import pickle

import streamlit as st

with open("churn-model.pck", "rb") as f:
    dv, model = pickle.load(f)

st.title("Predicción de Churn de Clientes")
st.header("Introduce los datos del cliente:")

contract = st.selectbox("Tipo de contrato", ["month-to-month", "one_year", "two_year"])
device_protection = st.selectbox(
    "Protección de dispositivos", ["no", "yes", "no_internet_service"]
)
internet_service = st.selectbox("Tipo de Internet", ["dsl", "fiber_optic", "no"])
monthly_charges = st.number_input(
    "Cargos mensuales", min_value=0.0, max_value=500.0, step=0.1
)
online_backup = st.selectbox(
    "Copia de seguridad online", ["no", "yes", "no_internet_service"]
)
online_security = st.selectbox("Seguridad online", ["no", "yes", "no_internet_service"])
payment_method = st.selectbox(
    "Método de pago",
    [
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
        "electronic_check",
        "mailed_check",
    ],
)
streaming_tv = st.selectbox(
    "Televisión en streaming", ["no", "yes", "no_internet_service"]
)
tech_support = st.selectbox("Soporte técnico", ["no", "yes", "no_internet_service"])
tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=100, step=1)
total_charges = st.number_input(
    "Cargos totales", min_value=0.0, max_value=10000.0, step=0.1
)

if st.button("Predecir"):
    client_data = {
        "contract": contract,
        "deviceprotection": device_protection,
        "internetservice": internet_service,
        "monthlycharges": monthly_charges,
        "onlinebackup": online_backup,
        "onlinesecurity": online_security,
        "paymentmethod": payment_method,
        "streamingtv": streaming_tv,
        "techsupport": tech_support,
        "tenure": tenure,
        "totalcharges": total_charges,
    }

    X_client = dv.transform([client_data])
    y_pred_proba = model.predict_proba(X_client)[0][1]

    st.subheader("Resultado:")
    if y_pred_proba > 0.5:
        st.error(f"El cliente tiene una alta probabilidad de churn: {y_pred_proba:.2f}")
    else:
        st.success(
            f"El cliente tiene una baja probabilidad de churn: {y_pred_proba:.2f}"
        )
