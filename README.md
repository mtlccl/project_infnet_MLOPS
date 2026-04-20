# 🏥 Insurance Cost Analysis - Pipeline MLOps Completo

Este projeto implementa uma arquitetura end-to-end de Machine Learning Operations (MLOps) para prever custos médicos individuais faturados por seguro de saúde, utilizando o **Medical Cost Personal Datasets** (Kaggle). 

O foco principal não é apenas a modelagem, mas a construção de uma infraestrutura robusta de **Treinamento Contínuo (CI/CD), Versionamento (Model Registry) e Monitoramento em Produção**.

---

## 🏗️ Arquitetura do Projeto

A solução foi construída utilizando as seguintes tecnologias:

* **Modelagem:** Scikit-Learn (Linear Regression, Ridge, GridSearchCV, Pipeline).
* **Tracking & Registry:** MLflow Server hospedado na nuvem.
* **Infraestrutura em Nuvem (AWS):** * **EC2:** Hospeda o Tracking Server do MLflow e atua como Runner do CI/CD.
  * **RDS (PostgreSQL):** *Backend Store* para armazenamento de metadados, hiperparâmetros e métricas de treinamento.
  * **S3:** *Artifact Store* para armazenamento dos binários do modelo (`.pkl`).
* **CI/CD:** GitHub Actions para retreinamento e deploy automatizado.
* **Front-end / Monitoramento:** Streamlit Cloud.

---

## 🎯 Mapeamento das Rubricas de Avaliação (Infnet)

### 1. Ambiente e Infraestrutura
* **[Rubrica 1.1] Arquitetura Integrada:** Implementada via Tracking Server remoto na AWS conectando código de treino, nuvem e Front-end (`app.py` e `experiment.py`).
* **[Rubrica 1.2] Armazenamento:** Uso configurado de S3 para artefatos e banco relacional RDS para métricas na instância EC2.
* **[Rubrica 1.3] Estrutura:** O projeto foi estruturado em uma organização de código adequada à prática de engenharia, reduzindo a dependência de notebooks e definindo fluxos claros, Isolamento total da responsabilidade de ingestão de dados..
* **[Rubrica 1.4] Reprodutibilidade:** Garantida no `experiment.py` pelo uso rigoroso de `random_state=42` no split e no pipeline fixo de preprocessamento.

### 2. Processamento de Dados e Engenharia de Features
* **[Rubrica 2.1] Ingestão e Estruturação:** Dados carregados de forma modular e isolada através da classe `DataLoaderDataset`, garantindo separação de responsabilidades no código.
* **[Rubrica 2.2] Pré-processamento Automatizado:** Construção de pipelines de transformação (`PreprocessorBuilder`) utilizando `StandardScaler` para padronização de numéricas (BMI, Age) e `OneHotEncoder` para variáveis categóricas (Smoker, Region, Sex).
* **[Rubrica 2.3] Redução de Dimensionalidade:** Implementação e teste de `PCA` integrado nativamente ao pipeline do Scikit-Learn, permitindo avaliar ganhos e perdas de explicabilidade no modelo.
* **[Rubrica 2.4] Avaliação de impacto:** Avaliação do impacto da redução de dimensionalidade (PCA)..

### 3. Experimentos e Modelagem
* **[Rubrica 3.1] Experimentos Claros:** Três cenários testados: `Linear_Baseline`, `Linear_PCA` e `Ridge_CV_Tuning` (`experiment.py`).
* **[Rubrica 3.2] Otimização (Tuning):** Uso do `GridSearchCV` para validação cruzada (CV=5) e busca do melhor `alpha` no modelo Ridge (`experiment.py`).
* **[Rubrica 3.3] Registros no MLflow:** Uso de `mlflow.log_param()`, `log_metric()` e `log_model()` em cada *run*.
* **[Rubrica 3.4] Decisão Técnica:** O modelo campeão (Ridge) foi mantido **sem PCA** para preservar a explicabilidade/transparência das variáveis categóricas (Fumante, BMI) que mostraram alto impacto na análise exploratória.

### 4. Deploy e Monitoramento
* **[Rubrica 4.1] Model Registry:** O modelo vencedor é automaticamente promovido para a "Vitrine" através da flag `registered_model_name="modelo_ridge"`.
* **[Rubrica 4.2] Endpoint de Inferência:** Criado via Streamlit (`page_prediction`), recebendo inputs não estruturados e retornando predições via requisição ao modelo da AWS.
* **[Rubrica 4.3] Geração de Métricas de Uso:** A cada inferência, o app gera e acopla metadados (Timestamp, Inputs, Confiança, RMSE dinâmico) à requisição.
* **[Rubrica 4.4] Dashboard de Observabilidade:** Interface em tempo real (`page_monitoring` no Streamlit) com gráficos Plotly exibindo Média Móvel, RMSE e Intervalos de Confiança para detectar degradação em produção.
* **[Rubrica 4.5] Usabilidade da Aplicação:** Interface profissional e fluida, construída como um *Multi-Page App*, com menu lateral, validação de inputs e Dark Mode configurado.

---

## 🚀 Como Executar o Projeto

### 1. Clonando o Repositório
```bash
git clone [https://github.com/mtlccl/MLOps_pjt_infnet.git](https://github.com/mtlccl/MLOps_pjt_infnet.git)
cd MLOps_pjt_infnet