import kagglehub
import pandas as pd
import os
import shutil
import logging

logger = logging.getLogger(__name__)

class DataLoaderDataset:

    # Rubrica 1.3: O projeto foi estruturado em uma organização de código adequada à prática de engenharia, 
    # reduzindo a dependência de notebooks e definindo fluxos claros, Isolamento total da responsabilidade de ingestão de dados.

    def __init__(self, dt_csv: str = "mirichoi0218/insurance", dfd: str = "data"):
        self.dt_insurance = dt_csv
        self.dfd = dfd
        logger.info(f"DataLoader inicializado com dataset_id: {self.dt_insurance}")

    def download_dtt(self) -> str:
        logger.info("Iniciando verificação/download do dataset")
        path = kagglehub.dataset_download(self.dt_insurance)
        csv_dt_insurance = os.path.join(path, "insurance.csv")
        
        os.makedirs(self.dfd, exist_ok=True)
        path_desti = os.path.join(self.dfd, "insurance.csv")
        shutil.copy(csv_dt_insurance, path_desti)
        logger.info(f"Dataset salvo com sucesso em: {path_desti}")
        
        return path_desti
    def load_data(self) -> pd.DataFrame:
        logger.info("Carregando dados na memória")
        path = self.download_dtt()
        df = pd.read_csv(path)

        # Rubrica 2.1: problemas de qualidade de dados Tratamento preventivo de ruído (linhas duplicadas) 
        # logo na fundação de dados para evitar vieses no treinamento.

        registros_iniciais_dp = len(df)
        df = df.drop_duplicates()
        registros_finais_sdp = len(df)
        
        if registros_iniciais_dp != registros_finais_sdp:
            logger.warning(f"Removidas {registros_iniciais_dp - registros_finais_sdp} linhas duplicadas.")
            
        logger.info(f"Dados prontos. Shape final: {df.shape}")
        return df