import pandas as pd
import streamlit as st

from OneRClassifier import OneRClassifier

# Configuração da página
st.set_page_config(
    page_title="Aprovação de Crédito - OneR", page_icon="💳", layout="wide"
)

# Título e descrição
st.title("Sistema de Aprovação de Crédito")
st.markdown("### Análise de crédito usando algoritmo OneR")


# Carrega e prepara os dados
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_labeled.csv")
    X = df.drop("Creditability", axis=1)
    y = df["Creditability"]

    # Treina o modelo
    classifier = OneRClassifier()
    classifier.fit(X, y)
    return df, X, y, classifier


df, X, y, classifier = load_data()

# Sidebar com informações do modelo
st.sidebar.header("Informações do Modelo")
st.sidebar.markdown(
    f"""
- **Feature mais relevante:** {X.columns[classifier.best_feature]}
- **Acurácia no treino:** {(sum(classifier.predict(X) == y) / len(y)):.2%}
"""
)

# Regras encontradas
st.sidebar.markdown("### Regras Descobertas")
for value, prediction in classifier.best_rule.items():
    st.sidebar.write(
        f"- Se *{X.columns[classifier.best_feature]}* = `{value}` → Creditability: `{prediction}`"
    )

# Área principal - Formulário de predição
st.markdown("## Análise de Novo Cliente")

# Criar colunas para organizar os campos
col1, col2 = st.columns(2)

with col1:
    account_balance = st.selectbox(
        "Saldo da Conta",
        options=["No account", "No balance", "Some balance"],
    )

    payment_status = st.selectbox(
        "Status de Pagamento de Créditos Anteriores",
        options=["No Problems", "Paid Up", "Some Problems"],
    )

    credit_amount = st.selectbox(
        "Valor do Crédito Atual", options=["Low", "Medium", "High"]
    )

    savings_value = st.selectbox(
        "Valor em Poupança/Ações", options=["None", "Below 50 EUR", "Above 50 EUR"]
    )

with col2:
    employment_length = st.selectbox(
        "Tempo no Emprego Atual",
        options=["Below 1 year", "1 to 4 years", "4 to 7 years", "Above 7 years"],
    )

    guarantors = st.selectbox("Garantidores", options=["None", "Yes"])

    concurrent_credits = st.selectbox(
        "Créditos Concorrentes", options=["None", "Other Banks or Dept Stores"]
    )

    num_credits = st.selectbox(
        "Número de Créditos neste Banco", options=["None", "One", "More than one"]
    )

# Botão para realizar a previsão
if st.button("Analisar Crédito", type="primary"):
    # Criar um DataFrame com os dados do formulário
    new_data = pd.DataFrame(
        {
            "Account_Balance": [account_balance],
            "Payment_Status_of_Previous_Credit": [payment_status],
            "Credit_Amount": [credit_amount],
            "Value_Savings_Stocks": [savings_value],
            "Length_of_current_employment": [employment_length],
            "Guarantors": [guarantors],
            "Concurrent_Credits": [concurrent_credits],
            "No_of_Credits_at_this_Bank": [num_credits],
        }
    )

    # Fazer a previsão
    prediction = classifier.predict(new_data)[0]

    # Exibir resultado
    st.markdown("### Resultado da Análise")

    if prediction == "Good":
        st.success("✅ Crédito Aprovado")
    else:
        st.error("❌ Crédito Negado")

    # Mostrar a regra que foi utilizada
    feature_value = new_data.iloc[0][X.columns[classifier.best_feature]]
    st.info(
        f"""
    **Regra aplicada:**
    - Se {X.columns[classifier.best_feature]} = {feature_value} → Creditability = {prediction}
    """
    )

# Dataset
st.markdown("## Dados de Treinamento")
st.dataframe(df, use_container_width=True)
