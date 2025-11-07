import os  
import numpy as np  
import pandas as pd 
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  

try:
    # Tentativa de importação padrão (se 'treino_modelo' for um pacote instalado ou no sys.path)
    from treino_modelo import treino
    from treino_modelo import pipelines
except Exception:
    # Se a importação direta falhar, usa 'importlib' para carregar os arquivos .py
    # localmente, com base no caminho do script atual.
    import importlib.util, os as _os
    # Carrega 'treino.py'
    spec = importlib.util.spec_from_file_location('treino', _os.path.join(_os.path.dirname(__file__), 'treino.py'))
    treino = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(treino)
    # Carrega 'pipelines.py'
    spec2 = importlib.util.spec_from_file_location('pipelines', _os.path.join(_os.path.dirname(__file__), 'pipelines.py'))
    pipelines = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(pipelines)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(BASE, 'analysis')
os.makedirs(OUT, exist_ok=True)


def evaluate_pipelines(n_splits=5):
    """
    Função para avaliar e comparar diferentes pipelines de ML usando validação cruzada de séries temporais.
    
    Args:
        n_splits (int): O número de "dobras" (folds) para o TimeSeriesSplit.
    """
    print('Carregando dados...')
    df = treino.load_data()
    # Pré-processa os dados 
    X, y, pre = treino.preprocess(df)

    # Construir pipelines
    # Cria as instâncias dos diferentes pipelines que serão testados
    
    # Pipeline 1: Baseado em modelo de árvore
    tree_pipe = pipelines.build_tree_pipeline(pre)
    # Pipeline 2: Modelo "sensível" com StandardScaler
    sens_pipe_std = pipelines.build_sensitive_pipeline(pre, scaler='standard', use_power=True)
    # Pipeline 3: Modelo "sensível" com RobustScaler (bom para outliers)
    sens_pipe_rob = pipelines.build_sensitive_pipeline(pre, scaler='robust', use_power=True)

    # Dicionário para iterar facilmente sobre os pipelines
    pipes = {'tree': tree_pipe, 'sensitive_std': sens_pipe_std, 'sensitive_robust': sens_pipe_rob}

    # Inicializa o objeto de validação cruzada para séries temporais
    # O TimeSeriesSplit garante que os dados de treino sempre venham antes dos dados de teste
    tss = TimeSeriesSplit(n_splits=n_splits)
    
    # Lista para armazenar os dicionários de resultados (métricas) de cada pipeline
    records = []

    # Itera sobre cada pipeline (nome e objeto)
    for name, pipe in pipes.items():
        # Listas para armazenar as métricas de cada "fold" (dobra da validação cruzada)
        maes, rmses, r2s = [], [], []
        print('Evaluando pipeline:', name)
        
        # O tss.split(X) gera os índices de treino e teste para cada dobra
        for train_idx, test_idx in tss.split(X):
            # Separa os dados de treino e teste com base nos índices
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # fit and predict (treinar e prever)
            try:
                # Treina o pipeline (incluindo pré-processamento e modelo)
                pipe.fit(X_train, y_train)
            except Exception as e:
                print('Error fitting', name, e) 
                continue  
            
            # Faz previsões nos dados de teste
            preds = pipe.predict(X_test)
            
            # Calcula as métricas de erro e armazena nas listas
            maes.append(mean_absolute_error(y_test, preds)) # Erro Absoluto Médio
            rmses.append(np.sqrt(mean_squared_error(y_test, preds))) # Raiz do Erro Quadrático Médio
            r2s.append(r2_score(y_test, preds)) # Coeficiente de Determinação R²

        # Após avaliar todas as dobras (folds), calcula a média das métricas para o pipeline atual
        records.append({
            'pipeline': name,
            'mae_mean': float(np.mean(maes)) if maes else np.nan,  # Média do MAE (se a lista não estiver vazia)
            'rmse_mean': float(np.mean(rmses)) if rmses else np.nan, # Média do RMSE
            'r2_mean': float(np.mean(r2s)) if r2s else np.nan  # Média do R²
        })

    # Converte a lista de resultados em um DataFrame do pandas e ordena pelo MAE
    out = pd.DataFrame(records).sort_values('mae_mean')
    
    # Salva o DataFrame de resultados em um arquivo CSV
    out.to_csv(os.path.join(OUT, 'pipeline_comparison.csv'), index=False, sep=';', decimal=',')
    print('Saved pipeline_comparison.csv') 
    
    print(out.to_string(index=False))

    return out


if __name__ == '__main__':
    evaluate_pipelines()
