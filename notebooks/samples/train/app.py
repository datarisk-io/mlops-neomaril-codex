import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import os


def train_model(base_path):
    """
    Função usada para treinar o modelo com base em um conjunto de dados fornecido.
    Essa função deve estruturar os passos que o Neomaril terá que executar para retornar o
    conjunto de informações resultantes do treino do modelo. 
    Na função você pode usar variáveis de ambiente carregadas a partir de um arquivo .env,
    como exemplificado no código nas linhas 50-53, e também é possível utilizar 
    uma variável de ambiente carregada diretamente pelo Neomaril (linhas 55-56) que é a:
    inputFileName : str
        Que contém o nome do arquivo da base de dados que foi feito upload


    Parâmetros
    ---------
    base_path : str
        O caminho de pastas para os arquivos que serão usados. Você pode usar um valor default para testes locais, mas no Neomaril será usado o caminho remoto dos arquivos.
        Por exemplo: "/path/to/treino/customizado/experimento1"

    Retorno
    -------
    dict: 
        um dicionário contendo as seguintes chaves
            X_train: DataFrame
                Os dados que serão usados para treinar o modelo
            y_train: DataFrame
                Dataframe/array/série de target que será usado para treinar o modelo
            model_output: DataFrame
                Dataframe/array/série com os resultados do modelo treinado. 
                Que podem ser os valores/probabilidades previstos, classes ou qualquer outra informação útil. 
                Essa informação precisa estar na saída do modelo implantado futuro para ser usada no monitoramento.
            pipeline: Pipeline
                A instância do modelo ajustado final. 
                Idealmente, deve ser uma classe Pipeline do Scikit-Learn, mas qualquer outra classe de algoritmo que implemente o método get_params funciona. 
                Isso será salvo como model.pkl com cloudpickle <https://github.com/cloudpipe/cloudpickle> ou com o método save_model se a classe de algoritmo tiver isso.
            metrics: dict
                Um dicionário com cada chave como um métrica. 
                Você pode usar qualquer nome para a chave da métrica e salvar quantos quiser, mas o valor deve ser numérico. 
                Por exemplo: {“auc_train”: 0,7, “auc_test”: 0,65}
            extra: string list
                Uma lista opcional de nomes de arquivos para arquivos extras que são gerados no treinamento. 
                Que podem ser gráficos, conjuntos de validação, etc. Eles precisam ser salvos no mesmo caminho (base_path) que é fornecido como parâmetro da função.
    """

    ## Você também pode usar variáveis de ambiente como nas linhas comentadas abaixo
    # my_var = os.getenv('MY_VAR')
    # if my_var is None:
    #    raise Exception("Could not find `env` variable")

    ## Variável de ambiente do Neomaril
    # df = pd.read_csv(base_path+'/'+os.getenv('inputFileName'))

    df = pd.read_csv(base_path+"/dados.csv")    # Carrega a base de dados, 'dados.csv' deve ser o nome do arquivo enviado para o Neomaril.
    # df = pd.read_csv(base_path+"/"+os.getenv('inputFileName'))    # Caso não queira deixar o nome do arquivo fixo o Neomaril salva o nome do arquivo enviado nessa variavel de ambiente
    
    # Os dados enviados devem ser os dados completos para treino e validação, então fica a critério do usuário como tratar os dados aqui
    X = df.drop(columns=['target'])             # Separa a base de dados da coluna com os targets
    y = df[["target"]]                          # Salva os targets num DataFrame
    
    pipe = make_pipeline(SimpleImputer(), LGBMClassifier())     # Define quais serão os passos para treinar o modelo
    
    # Nesse exemplo usamos a validação cruzada, mas isso fica a critério do usuário
    auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")  # Validação dos resultados usando a métrica 'auc'
    f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")   # Validação dos resultados usando a métrica 'f1'
    pipe.fit(X, y)  # Treina o modelo

    results = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})  # Constrói o DataFrame com os resultados
    
    # Retorna os resultados do treino segundo os parametros experados pelo Neomaril
    return {"X_train": X, "y_train": y, "model_output": results, "pipeline": pipe, 
            "metrics": {"auc": auc.mean(), "f1_score": f_score.mean()}}

