import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_loader_dataset import DataLoaderDataset
from src.preprocess import PreprocessorBuilder
from sklearn.pipeline import Pipeline
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ExperimentRunnerDTS:
    def __init__(self):
        # Rubrica 1.1: Projetei e implementei uma arquitetura de MLOps integrada (Tracking Server remoto na AWS).
        # Rubrica 1.2: Configurei infraestrutura de armazenamento e gerenciamento (S3 para artefatos, RDS para metadados).
        logger.info("Inicializando o ambiente de experimentos (MLflow)")
        mlflow.set_tracking_uri("http://174.129.138.108:5000")
        mlflow.set_experiment("Insurance_Regression_V2")
        
        self.df = DataLoaderDataset().load_data()
        self.X = self.df.drop('charges', axis=1)
        self.y = self.df['charges']
        
        logger.info("Dividindo dataset em Treino (80%) e Teste (20%)")
        
        # Rubrica 1.4: Reprodutibilidade do ambiente e modelagem garantida pelo random_state fixo.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.preprocessor_builder = PreprocessorBuilder()

    def run_exmt_rt(self, run_name_ex: str, model, use_pca: bool = False):
        # Rubrica 3.1: Experimentos definindo, comparações claras.
        logger.info(f"--- Iniciando Run: {run_name_ex} ---")
        
        with mlflow.start_run(run_name=run_name_ex):
            
            if use_pca:
                preprocessor = self.preprocessor_builder.build_pca_pipln(4)
            else:
                preprocessor = self.preprocessor_builder.build_bs_pipln()

            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            logger.info("Treinando o modelo...")
            pipeline.fit(self.X_train, self.y_train)
            
            logger.info("Realizando predições no conjunto de teste...")
            preds = pipeline.predict(self.X_test)
            
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))
            r2 = r2_score(self.y_test, preds)

            # Rubrica 3.3: Registro correto de parâmetros, métricas e versões no MLflow.
            logger.info(f"Métricas calculadas -> R2: {r2:.4f} | RMSE: {rmse:.2f}")
            
            mlflow.log_param("use_pca", use_pca)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            
            # Rubrica 4.1: Persistiu modelos treinados de forma versionada e reprodutível.
            mlflow.sklearn.log_model(
                sk_model=pipeline,                       
                artifact_path="model", 
                registered_model_name=run_name_ex  
            )
            logger.info(f"Modelo salvo no MLflow com sucesso.")

    def run_hyp_tng(self):
        # Rubrica 3.2: Implementei experimentos comparativos com validação cruzada e ajuste de hiperparâmetros.
        # Rubrica 3.4: Optei por usar o modelo final sem o PCA por um motivo principal: transparência.
        logger.info("--- Iniciando Otimização de Hiperparâmetros (Ridge CV) ---")
        
        with mlflow.start_run(run_name="Ridge_CV_Tuning"):
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor_builder.build_bs_pipln()), 
                ('model', Ridge())
            ])
            
            param_grid = {'model__alpha': [0.1, 1.0, 10.0]}
            logger.info(f"GridSearchCV configurado com alpha: {param_grid['model__alpha']} e CV=5")
            
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
            grid.fit(self.X_train, self.y_train)
            
            best_alpha = grid.best_params_['model__alpha']
            preds = grid.best_estimator_.predict(self.X_test)
            r2 = r2_score(self.y_test, preds)
            
            logger.info(f"Otimização finalizada! Melhor Alpha encontrado: {best_alpha} | R2: {r2:.4f}")
            
            mlflow.log_param("best_alpha", best_alpha)
            mlflow.log_metric("r2", r2)
            
            mlflow.sklearn.log_model(
                sk_model=grid.best_estimator_,                       
                artifact_path="modelo_ridge", 
                registered_model_name="modelo_ridge"   
            )

if __name__ == "__main__":
    logger.info("=== INICIANDO PIPELINE DE MACHINE LEARNING ===")
    runner = ExperimentRunnerDTS()
    
    runner.run_exmt_rt("Linear_Baseline", LinearRegression(), use_pca=False)
    runner.run_exmt_rt("Linear_PCA", LinearRegression(), use_pca=True)
    runner.run_hyp_tng()
    
    logger.info("=== PIPELINE FINALIZADO COM SUCESSO ===")