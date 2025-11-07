import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import joblib
from scipy.stats import loguniform, uniform

# Importar funções do treino.py
from treino import load_data, preprocess

def create_param_space():
    """Define o espaço de busca dos hiperparâmetros"""
    return {
        'regressor__alpha': loguniform(1e-5, 1.0),      # força da regularização (log-uniforme para cobrir várias ordens de magnitude)
        'regressor__epsilon': uniform(1.1, 2.0),         # robustez a outliers (entre 1.1 e 3.1)
        'regressor__max_iter': [100, 200, 500, 1000],    # max iterações
        'regressor__warm_start': [True, False],          # warm start pode ajudar na convergência
        'regressor__tol': loguniform(1e-5, 1e-3)        # tolerância para convergência
    }

def evaluate_model(y_true, y_pred):
    """Calcula múltiplas métricas de avaliação"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

def tune_huber(n_trials=50, n_splits=5):
    print('Loading data...')
    df = load_data()
    print('Rows:', len(df))

    # Preprocessamento padrão do treino.py
    X, y, preprocessor = preprocess(df)
    
    # Pipeline base
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', HuberRegressor())
    ])

    # Configurar validação cruzada temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Definir múltiplos scorers
    scorers = {
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'rmse': make_scorer(lambda y, p: np.sqrt(mean_squared_error(y, p)), greater_is_better=False),
        'r2': make_scorer(r2_score, greater_is_better=True)
    }

    # Configurar RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=create_param_space(),
        n_iter=n_trials,
        cv=tscv,
        scoring=scorers,
        refit='mae',  # otimizar para MAE
        random_state=42,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    print(f'\nIniciando busca com {n_trials} tentativas...')
    random_search.fit(X, y)

    # Resultados
    print('\nMelhores hiperparâmetros encontrados:')
    for param, value in random_search.best_params_.items():
        print(f'{param}: {value}')

    print('\nMelhores scores:')
    print(f'MAE: {-random_search.cv_results_["mean_test_mae"][random_search.best_index_]:,.2f}')
    print(f'RMSE: {-random_search.cv_results_["mean_test_rmse"][random_search.best_index_]:,.2f}')
    print(f'R²: {random_search.cv_results_["mean_test_r2"][random_search.best_index_]:.4f}')

    # Treinar modelo final com os melhores parâmetros
    print('\nTreinando modelo final com os melhores parâmetros...')
    
    # Divisão cronológica final
    df['MesAno_dt'] = pd.to_datetime(df['MesAno'] + '-01', errors='coerce')
    df = df.sort_values('MesAno_dt')
    split_idx = int(len(df) * 0.8)
    train_mask = df.index < df.index[split_idx]
    test_mask = ~train_mask

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)
    
    # Avaliar no conjunto de teste
    preds = best_model.predict(X_test)
    final_metrics = evaluate_model(y_test, preds)
    
    print('\nMétricas finais (holdout):')
    print(f"MAE: {final_metrics['mae']:,.2f}")
    print(f"RMSE: {final_metrics['rmse']:,.2f}")
    print(f"R²: {final_metrics['r2']:.4f}")

    # Salvar modelo e resultados
    model_path = os.path.join(os.path.dirname(__file__), 'modelos', 'huber_model_tuned.joblib')
    joblib.dump(best_model, model_path)
    print(f'\nModelo salvo em {model_path}')

    # Gerar e salvar previsões
    df.loc[test_mask, 'Pred_Consumo'] = preds
    predictions_path = os.path.join(os.path.dirname(__file__), 'modelos', 'predictions_huber_tuned.csv')
    df.loc[test_mask, ['MesAno', 'Regiao', 'Consumo', 'Pred_Consumo']].to_csv(
        predictions_path, index=False, sep=';', decimal=','
    )
    print(f'Previsões salvas em {predictions_path}')

    # Atualizar resultados comparativos
    results_path = os.path.join(os.path.dirname(__file__), 'resultados_modelos_comparacao.csv')
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, sep=';', decimal=',')
        # Criar nova linha com resultados do modelo tunado
        new_row = pd.DataFrame([{
            'model': 'huber_tuned',
            'cv_mae': -random_search.cv_results_['mean_test_mae'][random_search.best_index_],
            'cv_rmse': -random_search.cv_results_['mean_test_rmse'][random_search.best_index_],
            'cv_r2': random_search.cv_results_['mean_test_r2'][random_search.best_index_],
            'hold_mae': final_metrics['mae'],
            'hold_rmse': final_metrics['rmse'],
            'hold_r2': final_metrics['r2']
        }])
        
        # Remove linha anterior do huber_tuned se existir
        results_df = results_df[results_df['model'] != 'huber_tuned']
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(results_path, index=False, sep=';', decimal=',')
        print(f'\nResultados atualizados em {results_path}')

    return random_search

if __name__ == '__main__':
    tune_huber(n_trials=50)  # 50 tentativas de diferentes combinações de hiperparâmetros