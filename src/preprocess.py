import sklearn.pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class PreprocessorBuilder:
    def __init__(self):
        # Rubrica 2.2: Identificação de tipos de dados e separação entre variáveis numéricas e categóricas.
        self.num_features = ['age', 'bmi', 'children']
        self.cat_features = ['sex', 'smoker', 'region']
        logger.info("PreprocessorBuilder inicializado.")

    def build_bs_pipln(self) -> ColumnTransformer:
        # Rubrica 2.3: Implementação de Engenharia de Atributos (Scaling para numéricas e OneHot para categóricas).
        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num_features),
                ('cat', cat_transformer, self.cat_features)
            ]
        )
        logger.info("Construindo pipeline base (StandardScaler + OneHotEncoder)...")
        return preprocessor

    def build_pca_pipln(self, n_components: int) -> Pipeline:
        # Rubrica 2.4: Avaliação do impacto da redução de dimensionalidade (PCA).
        logger.info(f"Construindo pipeline com redução de dimensionalidade PCA (n_components={n_components})...")
        
        pca_pipeline = Pipeline(steps=[
            ('preprocessor', self.build_bs_pipln()),
            ('pca', PCA(n_components=n_components))
        ])
        return pca_pipeline