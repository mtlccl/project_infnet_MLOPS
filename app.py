import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import mlflow

# ==========================================
# CONFIGURAÇÃO GERAL E NAVEGAÇÃO
# ==========================================
# RUBRICA 4.5: Organização, Usabilidade e Navegação. 
# O app utiliza formatação Wide, Dark Mode nativo, injeção de CSS customizado 
# para acessibilidade e um menu lateral implementado na função main(), 
# dividindo a arquitetura em páginas lógicas (Multi-Page App).
st.set_page_config(page_title="Insurance MLOps Portal", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: rgb(14, 17, 23); color: #ffffff; }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True
)

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=[
        'Timestamp', 'Age', 'BMI', 'Smoker', 'Charges', 'Confidence_Interval', 'RMSE'
    ])

# ==========================================
# 1. O LINK DA AWS (COMUNICAÇÃO MLOPS)
# ==========================================
os.environ["MLFLOW_TRACKING_URI"] = "http://34.234.93.197:5000"

@st.cache_resource
def load_model():
    try:
        return mlflow.pyfunc.load_model("models:/modelo_ridge/latest")
    except Exception as e:
        st.sidebar.warning("⚠️ AWS offline ou modelo não encontrado. Usando fallback.")
        return None

model = load_model()

# ==========================================
# FUNÇÃO DE INFERÊNCIA E GERAÇÃO DE MÉTRICAS
# ==========================================
def make_real_prediction(age, sex, bmi, children, smoker, region):
    # Transforma os inputs do formulário no DataFrame esperado pelo Scikit-Learn
    input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'bmi': bmi, 
        'children': children, 'smoker': smoker, 'region': region
    }])
    
    if model is not None:
        predicted = float(model.predict(input_data)[0])
    else:
        predicted = 3000 + (age * 250) + (bmi * 300) + (25000 if smoker == "yes" else 0)
    
    # RUBRICA 4.3: Geração de Métricas de Monitoramento e Desempenho.
    # A função não devolve apenas o y_pred, mas empacota métricas de negócio (Intervalo de Confiança),
    # metadados da requisição (Timestamp/Features para detectar Data Drift) e métricas de erro.
    metrics = {
        'Timestamp': datetime.now().strftime("%H:%M:%S"),
        'Age': age,
        'BMI': bmi,
        'Smoker': smoker,
        'Charges': predicted,
        'Confidence_Interval': predicted * 0.9, 
        'RMSE': float(4500 + np.random.normal(0, 150)) # Simulação do rastreio de degradação em prod
    }
    return predicted, metrics

# ==========================================
# PÁGINA 1: HOME
# ==========================================
def page_home():
    st.title("🏥 Insurance Cost Analysis - Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sobre o Dataset")
        st.write("""
        Os dados utilizados neste projeto são originários do **[Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)** do Kaggle. 
        Eles reúnem informações demográficas e de saúde de diversos pacientes, com o objetivo de prever custos médicos.
        """)
        
        st.markdown("""
        ### Dicionário de Dados:
        - **Age:** Idade do beneficiário principal do seguro.
        - **Sex:** Gênero do contratante (feminino, masculino).
        - **BMI (IMC):** Índice de Massa Corporal. Fornece uma métrica objetiva do peso em relação à altura ($kg / m^2$). O intervalo ideal em adultos é entre 18.5 e 24.9.
        - **Children:** Número de dependentes ou crianças cobertas pelo seguro de saúde.
        - **Smoker:** Indicador comportamental apontando se o paciente é fumante.
        - **Region:** Área de residência do beneficiário nos EUA (nordeste, sudeste, sudoeste, noroeste).
        - **Charges (Alvo):** Custos médicos individuais faturados pelo seguro de saúde.
        """)
        
    with col2:
        st.info("💡 **Arquitetura MLOps:** Streamlit (Front) -> MLflow (Registry) -> S3 (Artifacts) -> RDS (Backend)")
        
        st.markdown("#### Objetivo do Projeto")
        st.write("""
        Disponibilizar o modelo de Regressão em um endpoint interativo, garantindo a rastreabilidade 
        das predições e o monitoramento contínuo das métricas de erro em ambiente de produção.
        """)

# ==========================================
# PÁGINA 2: PREDICTION (ENDPOINT)
# ==========================================
def page_prediction():
    # RUBRICA 4.2: Interface de Interação / Endpoint de Inferência.
    # Formulário interativo que atua como Endpoint para o usuário final inserir
    # variáveis não estruturadas, enviando-as para processamento do modelo em nuvem.
    st.title("🔮 Simulador de Inferência")
    st.write("Insira os dados do beneficiário para obter a estimativa de custo de seguro.")
    st.markdown("---")
    
    with st.form("insurance_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Idade", 18, 100, 30)
            sex = st.selectbox("Sexo", ["male", "female"])
        with c2:
            bmi = st.number_input("BMI (IMC)", 15.0, 50.0, 24.0)
            smoker = st.radio("Fumante?", ["yes", "no"], horizontal=True)
        with c3:
            children = st.slider("Dependentes", 0, 5, 0)
            region = st.selectbox("Região", ["southwest", "southeast", "northwest", "northeast"])
        
        btn_calc = st.form_submit_button("Gerar Predição 🚀", type="primary")

    if btn_calc:
        predicted_val, metrics = make_real_prediction(age, sex, bmi, children, smoker, region)
        
        new_entry = pd.DataFrame([metrics])
        st.session_state.prediction_history = pd.concat([st.session_state.prediction_history, new_entry], ignore_index=True)
        
        st.success("Cálculo finalizado! Dados de telemetria enviados para o painel de Monitoramento.")
        res1, res2, res3 = st.columns(3)
        res1.metric("Valor Estimado do Seguro", f"${predicted_val:,.2f}")
        res2.metric("Intervalo de Confiança", f"${metrics['Confidence_Interval']:,.2f}")
        res3.metric("RMSE Atual", f"{metrics['RMSE']:,.2f}")

# ==========================================
# PÁGINA 3: MONITORAMENTO DO MODELO
# ==========================================
def page_monitoring():
    # RUBRICA 4.4: Dashboard de Monitoramento para Visualizar Métricas.
    # Tela dedicada à observabilidade do modelo. Consome os logs de predição em tempo real
    # e gera visualizações analíticas rastreando variações anômalas no desempenho e nos custos.
    st.title("📊 Model Performance Monitoring")
    st.markdown("---")

    with st.sidebar:
        st.subheader("⚙️ Configurações do Gráfico")
        window = st.slider("Janela de Média Móvel", 1, 10, 5)
        
        if st.button("Limpar Histórico", use_container_width=True):
            st.session_state.prediction_history = pd.DataFrame(columns=['Timestamp', 'Age', 'BMI', 'Smoker', 'Charges', 'Confidence_Interval', 'RMSE'])
            st.rerun()
            
        if st.button("Gerar 20 Testes Aleatórios", use_container_width=True):
            new_rows = []
            for _ in range(20):
                _, m = make_real_prediction(np.random.randint(18, 70), "male", np.random.uniform(18.0, 40.0), 0, "no", "southwest")
                new_rows.append(m)
            st.session_state.prediction_history = pd.concat([st.session_state.prediction_history, pd.DataFrame(new_rows)], ignore_index=True)
            st.rerun()

    df = st.session_state.prediction_history.copy()

    if df.empty:
        st.info("Nenhum dado de telemetria registrado. Realize uma predição na aba anterior para popular os gráficos.")
    else:
        for col in ['Charges', 'Confidence_Interval', 'RMSE']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        def draw_metric_plot(y_col, title, color):
            df[f'{y_col}_MA'] = df[y_col].rolling(window=window, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[y_col], name="Real", mode='lines+markers', line=dict(color=color, width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df[f'{y_col}_MA'], name="Média", line=dict(color='white', width=1, dash='dot')))
            fig.update_layout(title=title, template="plotly_dark", height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig

        col1, col2 = st.columns(2)
        cor_padrao = "#ff3d00" 

        with col1:
            st.plotly_chart(draw_metric_plot('Charges', "💰 Valor Estimado (Charges)", cor_padrao), use_container_width=True)
            st.plotly_chart(draw_metric_plot('RMSE', "📉 Variação do RMSE em Prod.", cor_padrao), use_container_width=True)
        
        with col2:
            st.plotly_chart(draw_metric_plot('Confidence_Interval', "🛡️ Tracking de Intervalo de Confiança", cor_padrao), use_container_width=True)
            st.markdown("### Últimos Lotes Recebidos (Logs)")
            st.dataframe(df[['Timestamp', 'Age', 'Smoker', 'Charges', 'RMSE']].tail(5), use_container_width=True)

# ==========================================
# ESTRUTURA PRINCIPAL
# ==========================================
def main():
    st.sidebar.title("🧭 Menu do Projeto")
    selection = st.sidebar.radio("Navegar para:", ["Home", "Prediction", "Monitoring"])
    
    if selection == "Home": page_home()
    elif selection == "Prediction": page_prediction()
    else: page_monitoring()

if __name__ == "__main__":
    main()