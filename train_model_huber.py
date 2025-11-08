import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# Importar funções do treino.py
from treino import load_data, preprocess, recommend_sustainable

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'modelos')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_huber(n_splits=5):
    print('Loading data...')
    df = load_data()
    print('Rows:', len(df))

    # Usar o mesmo preprocessamento do treino.py
    X, y, preprocessor = preprocess(df)
    
    # Configuração do HuberRegressor
    huber = HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=100, warm_start=False)

    # Pipeline completa
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', huber)
    ])

    # Avaliação com TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=n_splits)
    cv_maes = []
    cv_rmses = []
    cv_r2s = []

    print('Evaluating with TimeSeriesSplit...')
    for train_idx, test_idx in tss.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        cv_maes.append(mean_absolute_error(y_test, preds))
        cv_rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
        cv_r2s.append(r2_score(y_test, preds))

    # Métricas da validação cruzada
    cv_mae = np.mean(cv_maes)
    cv_rmse = np.mean(cv_rmses)
    cv_r2 = np.mean(cv_r2s)

    print('\nCross-validation metrics:')
    print(f'CV MAE: {cv_mae:,.2f}')
    print(f'CV RMSE: {cv_rmse:,.2f}')
    print(f'CV R²: {cv_r2:.4f}')

    # Treino final com holdout cronológico
    df['MesAno_dt'] = pd.to_datetime(df['MesAno'] + '-01', errors='coerce')
    df = df.sort_values('MesAno_dt')
    split_idx = int(len(df) * 0.8)
    train_mask = df.index < df.index[split_idx]
    test_mask = ~train_mask

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print('\nTraining final model...')
    pipe.fit(X_train, y_train)

    # Avaliação holdout
    preds = pipe.predict(X_test)
    hold_mae = mean_absolute_error(y_test, preds)
    hold_rmse = np.sqrt(mean_squared_error(y_test, preds))
    hold_r2 = r2_score(y_test, preds)

    print('\nHoldout metrics:')
    print(f'Holdout MAE: {hold_mae:,.2f}')
    print(f'Holdout RMSE: {hold_rmse:,.2f}')
    print(f'Holdout R²: {hold_r2:.4f}')

    # Salvar modelo
    model_path = os.path.join(MODEL_DIR, 'huber_model.joblib')
    joblib.dump(pipe, model_path)
    print(f'\nModel saved to {model_path}')

    # Gerar recomendações e salvar previsões
    df.loc[test_mask, 'Pred_Consumo'] = preds
    df['Recomendacao'] = df.apply(recommend_sustainable, axis=1)

    # Salvar previsões do conjunto de teste
    predictions_path = os.path.join(MODEL_DIR, 'predictions_huber.csv')
    df.loc[test_mask, ['MesAno', 'Regiao', 'Consumo', 'Pred_Consumo']].to_csv(
        predictions_path, index=False, sep=';', decimal=','
    )
    print(f'Test predictions saved to {predictions_path}')

    # Salvar amostra com recomendações
    cols_amostra = ['MesAno', 'Regiao', 'Estacao', 'Consumo', 
                    'TEMPERATURA MEDIA, MENSAL (AUT)(°C)',
                    'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)',
                    'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)',
                    'Recomendacao']
    
    sample_path = os.path.join(MODEL_DIR, 'recomendacoes_amostra_huber.csv')
    df[cols_amostra].to_csv(sample_path, index=False, sep=';', decimal=',')
    print(f'Recommendations sample saved to {sample_path}')

    # Atualizar resultados comparativos
    results_path = os.path.join(BASE, 'resultados_modelos_comparacao.csv')
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, sep=';', decimal=',')
        # Atualizar ou adicionar linha do Huber
        new_row = pd.DataFrame([{
            'model': 'huber',
            'cv_mae': cv_mae,
            'cv_rmse': cv_rmse,
            'cv_r2': cv_r2,
            'hold_mae': hold_mae,
            'hold_rmse': hold_rmse,
            'hold_r2': hold_r2
        }])
        
        # Remove linha anterior do huber se existir
        results_df = results_df[results_df['model'] != 'huber']
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(results_path, index=False, sep=';', decimal=',')
        print(f'\nResults updated in {results_path}')

if __name__ == '__main__':
    train_huber()