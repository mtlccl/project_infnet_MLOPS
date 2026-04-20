import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px
from mlflow.tracking import MlflowClient

# ==========================================
# CONFIGURAÇÕES GERAIS E CONEXÃO AWS
# ==========================================
MLFLOW_TRACKING_URI = "http://174.129.138.108:5000"
EXPERIMENT_NAME = "Insurance_Regression_V2"
MODEL_NAME = "modelo_ridge"

# Aponta o Streamlit para o servidor remoto do MLflow na AWS
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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
# PÁGINA 2: PREDIÇÃO (SIMULADOR)
# ==========================================
# Rubrica 4.2: Endpoint de Inferência (Recebe inputs não estruturados e retorna predições da AWS)
def page_prediction():
    st.title("🔮 Simulador de Custos Médicos")
    st.markdown("Preencha os dados do paciente para prever os custos faturados pelo seguro.")
    st.markdown("---")
    
    # Formulário de entrada dos dados do paciente
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Idade (Age)", min_value=18, max_value=100, value=30)
        bmi = st.number_input("IMC (BMI)", min_value=10.0, max_value=50.0, value=25.0, format="%.1f")
        children = st.number_input("Dependentes (Children)", min_value=0, max_value=10, value=0)
    with col2:
        sex = st.selectbox("Gênero (Sex)", ["male", "female"])
        smoker = st.selectbox("Fumante (Smoker)", ["yes", "no"])
        region = st.selectbox("Região (Region)", ["northeast", "northwest", "southeast", "southwest"])
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Calcular Previsão", type="primary", use_container_width=True):
        try:
            with st.spinner("Conectando à AWS e carregando o modelo mais recente..."):
                model_uri = f"models:/{MODEL_NAME}/latest"
                model = mlflow.pyfunc.load_model(model_uri)
            
            # Monta os dados exatamente no formato que o pipeline treinado espera
            input_data = pd.DataFrame({
                "age": [age],
                "sex": [sex],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "region": [region]
            })
            
            # Rubrica 4.3: Geração de predição em tempo real e acoplamento de inputs/outputs
            prediction = model.predict(input_data)[0]
            
            st.success("Predição realizada com sucesso pela nuvem!")
            st.metric(label="Custo Médico Previsto", value=f"US$ {prediction:,.2f}")
            
        except Exception as e:
            st.error(f"Erro ao realizar a predição: {e}")
            st.info("Verifique se o modelo 'modelo_ridge' está registrado corretamente no MLflow.")

# ==========================================
# PÁGINA 3: MONITORAMENTO
# ==========================================
def get_mlflow_data():
    """Busca o histórico de execuções do MLflow na AWS"""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            st.error(f"Experimento '{EXPERIMENT_NAME}' não encontrado no servidor AWS.")
            return pd.DataFrame()
            
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return runs
    except Exception as e:
        st.error(f"Erro ao conectar ao Tracking Server: {e}")
        return pd.DataFrame()

# Rubrica 4.4: Dashboard de Observabilidade (Interface em tempo real exibindo RMSE e R2)
def page_monitoring():
    st.title("📊 Monitoramento de Modelos (MLOps)")
    st.markdown("Acompanhe a evolução das métricas de treinamento diretamente do servidor AWS.")
    st.markdown("---")
    
    with st.spinner("Buscando logs de treinamento na AWS..."):
        df_runs = get_mlflow_data()
    
    if df_runs.empty:
        st.warning("Nenhum dado de execução encontrado no MLflow para gerar os gráficos.")
    else:
        # Prepara a linha do tempo
        df_runs['start_time'] = pd.to_datetime(df_runs['start_time'])
        df_runs = df_runs.sort_values('start_time')

        # Nomes exatos das colunas geradas pelo MLflow
        col_rmse = "metrics.rmse"
        col_r2 = "metrics.r2"

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Evolução do Erro (RMSE)")
            if col_rmse in df_runs.columns:
                fig_rmse = px.line(df_runs, x='start_time', y=col_rmse, 
                                 markers=True, title="RMSE por Treinamento (Menor é melhor)")
                st.plotly_chart(fig_rmse, use_container_width=True)
            else:
                st.error("Métrica 'rmse' não encontrada nos logs da AWS.")

        with col2:
            st.subheader("Qualidade do Modelo (R²)")
            if col_r2 in df_runs.columns:
                fig_r2 = px.area(df_runs, x='start_time', y=col_r2, 
                                title="R² Score por Treinamento (Maior é melhor)")
                st.plotly_chart(fig_r2, use_container_width=True)
            else:
                st.error("Métrica 'r2' não encontrada nos logs da AWS.")

        st.markdown("### Histórico Bruto (Runs)")
        
        # Filtra apenas as colunas que existem para não quebrar a tabela
        cols_to_show = ['run_id', 'start_time']
        if col_r2 in df_runs.columns: cols_to_show.append(col_r2)
        if col_rmse in df_runs.columns: cols_to_show.append(col_rmse)
        
        st.dataframe(df_runs[cols_to_show].dropna())

# ==========================================
# MENU LATERAL E NAVEGAÇÃO PRINCIPAL
# ==========================================
# Rubrica 4.5: Usabilidade da Aplicação (App multi-page, estruturado com menu e validações visuais)
def main():
    st.set_page_config(page_title="Insurance MLOps", page_icon="🏥", layout="wide")
    
    st.sidebar.title("Menu MLOps")
    st.sidebar.markdown("Navegue pelas etapas do projeto:")
    
    selection = st.sidebar.radio("Selecione a página:", 
                                 ["🏠 Home", "🔮 Predição (Simulador)", "📊 Monitoramento"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Conectado em:**\n{MLFLOW_TRACKING_URI}")
    
    if selection == "🏠 Home":
        page_home()
    elif selection == "🔮 Predição (Simulador)":
        page_prediction()
    elif selection == "📊 Monitoramento":
        page_monitoring()

if __name__ == "__main__":
    main()
