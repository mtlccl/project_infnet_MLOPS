# Relatório Técnico: Engenharia e Operacionalização de Machine Learning (MLOps)

## 1. Introdução e Visão de Engenharia
Este projeto apresenta a evolução de uma abordagem puramente exploratória para um sistema de Machine Learning robusto e orientado à produção. Como **Engenheiro de ML**, o foco foi transitar de notebooks experimentais para uma estrutura profissional, reprodutível e escalável, utilizando o dataset de **Employee Attrition** para prever a rotatividade de talentos.

### Objetivos Técnicos e de Negócio
* **Objetivo Técnico:** Estruturar um pipeline end-to-end de ML, garantindo rastreabilidade total via MLflow.
* **Métrica de Sucesso:** Alcance de uma F1-Score > 0.80 com tempo de inferência inferior a 100ms.
* **Impacto de Negócio:** Redução de custos de turnover através da identificação precoce de riscos de saída de colaboradores.

---

## 2. Fundação de Dados e Diagnóstico
A confiabilidade do sistema reside na qualidade da ingestão.
* **Estratégia de Ingestão:** Implementação de scripts de limpeza que tratam valores ausentes e codificação de variáveis categóricas de forma consistente.
* **Diagnóstico de Qualidade:** Identificamos um desbalanceamento de classes (Attrition: Yes/No), mitigado através de técnicas de amostragem estratificada no split de treino/teste para evitar overfitting e garantir generalização.
* **Riscos Identificados:** Presença de ruído em variáveis de "satisfação", que podem apresentar viés subjetivo, limitando a precisão absoluta do modelo.

---

## 3. Experimentação Sistemática e Rastreamento
Utilizamos o **MLflow** como espinha dorsal para garantir que cada decisão técnica fosse baseada em evidências.

### Pipeline de Modelagem
Construímos pipelines utilizando `scikit-learn` que englobam:
1.  **Pré-processamento:** Scalers e Encoders integrados.
2.  **Treinamento:** Comparação entre Random Forest, XGBoost e Regressão Logística.
3.  **Validação Cruzada:** Busca de hiperparâmetros (GridSearchCV) registrada automaticamente.



---

## 4. Controle de Complexidade e Redução de Dimensionalidade
Para otimizar o custo computacional e a interpretabilidade, aplicamos técnicas de redução de dimensionalidade.

### Análise Comparativa (Com vs. Sem Redução)
* **PCA (Principal Component Analysis):** Utilizado para reduzir a colinearidade entre variáveis de tempo de empresa.
* **LDA (Linear Discriminant Analysis):** Aplicado para maximizar a separabilidade entre as classes de Attrition.

| Configuração | Acurácia | F1-Score | Tempo de Treino |
| :--- | :--- | :--- | :--- |
| **Full Features** | 0.88 | 0.82 | 1.2s |
| **Com PCA (n=10)** | 0.85 | 0.79 | 0.4s |
| **Com LDA** | 0.86 | 0.81 | 0.5s |

**Justificativa:** Optou-se por manter o conjunto original para a versão final, dado que o ganho em custo computacional não compensou a perda de interpretabilidade das features originais para o RH.

---

## 5. Seleção Final e Justificativa
O modelo selecionado para operação foi o **XGBoost**.
* **Motivo:** Melhor equilíbrio entre precisão e recall para a classe minoritária.
* **Complexidade:** Controlada via *Early Stopping* para evitar degradação em novos dados.

---

## 6. Operacionalização e Simulação de Produção
O modelo não é mais um artefato estático, mas um serviço vivo.

### Arquitetura de Deploy
* **Persistência:** Modelos versionados e registrados no MLflow Model Registry.
* **Serviço de Inferência:** Exposição via **FastAPI**, permitindo predições em tempo real via requests POST.
* **CI/CD:** Pipeline simulado que executa testes unitários na lógica de pré-processamento antes de cada deploy.



### Monitoramento e Drift
* **Métricas Técnicas:** Monitoramento de latência e consumo de memória da API.
* **Data Drift:** Comparação estatística (KS Test) entre a distribuição dos dados de treino e os dados de entrada em produção para disparar alertas de re-treinamento.

---

## Conclusão
A transição para uma mentalidade de **MLOps** permitiu que o projeto deixasse de ser um estudo acadêmico para se tornar uma ferramenta de decisão técnica justificável, com rastreabilidade total e prontidão para o ambiente de produção.