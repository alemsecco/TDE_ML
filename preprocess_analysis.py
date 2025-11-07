import os
import pandas as pd
import numpy as np
from scipy import stats
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

try:
    from treino_modelo import treino
except Exception:
    import importlib.util, os as _os
    spec = importlib.util.spec_from_file_location('treino', _os.path.join(_os.path.dirname(__file__), 'treino.py'))
    treino = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(treino)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)


def run_analysis():
    print('Carregando dados...')
    df = treino.load_data()
    print('Linhas:', len(df))

    # certificar Consumo numerico
    df = df.copy()
    if 'Consumo' in df.columns:
        df['Consumo'] = pd.to_numeric(df['Consumo'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')

    # identificar colunas numéricas 
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ('Ano', 'Mes')]
    # se algumas colunas numéricas estão como object, tentar coagir as comuns
    for c in df.columns:
        if c not in numeric_cols and df[c].dtype == object:
            # tentar converter heuristicamente
            coerced = pd.to_numeric(df[c].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
            if coerced.notna().sum() > 0:
                numeric_cols.append(c)
                df[c] = coerced

    print('Numeric cols detected:', numeric_cols)

    # stats descritivas
    desc = df[numeric_cols].describe().T
    desc['skew'] = df[numeric_cols].skew()
    desc['kurtosis'] = df[numeric_cols].kurtosis()
    desc.to_csv(os.path.join(OUT_DIR, 'numeric_describe.csv'), sep=';', decimal=',')
    print('Saved numeric_describe.csv')

    # Correlações
    corr_pearson = df[numeric_cols].corr(method='pearson')
    corr_spearman = df[numeric_cols].corr(method='spearman')
    corr_pearson.to_csv(os.path.join(OUT_DIR, 'corr_pearson.csv'), sep=';', decimal=',')
    corr_spearman.to_csv(os.path.join(OUT_DIR, 'corr_spearman.csv'), sep=';', decimal=',')
    print('Saved correlation matrices')

    # Heatmaps (pequenos, se mtas colunas grande)
    if PLOTTING_AVAILABLE:
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_pearson, cmap='RdBu_r', center=0)
            plt.title('Pearson correlation')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, 'heatmap_pearson.png'))
            plt.close()

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_spearman, cmap='RdBu_r', center=0)
            plt.title('Spearman correlation')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, 'heatmap_spearman.png'))
            plt.close()
            print('Saved heatmaps')
        except Exception as e:
            print('Could not save heatmaps:', e)
    else:
        print('Plotting libs not available; skipping heatmaps.')

    # detecção de Outlier: IQR por feature (global) e por region
    iqr_rows = []
    for c in numeric_cols:
        series = df[c].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = ((series < lower) | (series > upper)).sum()
        iqr_rows.append({'feature': c, 'q1': q1, 'q3': q3, 'iqr': iqr, 'lower': lower, 'upper': upper, 'n_outliers': int(n_out), 'total': int(len(series))})
    iqr_df = pd.DataFrame(iqr_rows)
    iqr_df.to_csv(os.path.join(OUT_DIR, 'iqr_outliers_global.csv'), sep=';', decimal=',')
    print('Saved iqr_outliers_global.csv')

    # IQR por region (contagem)
    if 'Regiao' in df.columns:
        regions = df['Regiao'].dropna().unique()
        rows = []
        for reg in regions:
            sub = df[df['Regiao'] == reg]
            for c in numeric_cols:
                s = sub[c].dropna()
                if s.empty:
                    continue
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_out = ((s < lower) | (s > upper)).sum()
                rows.append({'Regiao': reg, 'feature': c, 'n': len(s), 'n_outliers': int(n_out)})
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'iqr_outliers_by_region.csv'), sep=';', decimal=',')
        print('Saved iqr_outliers_by_region.csv')

    # IsolationForest para anomalias multivariadas nas features numéricas
    try:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        Xnum = df[numeric_cols].fillna(df[numeric_cols].median())
        iso.fit(Xnum)
        scores = iso.decision_function(Xnum)
        is_out = iso.predict(Xnum) == -1
        df_iso = pd.DataFrame({'score': scores, 'is_outlier': is_out})
        df_iso.to_csv(os.path.join(OUT_DIR, 'isolationforest_scores.csv'), sep=';', decimal=',')
        print('Saved isolationforest_scores.csv')
    except Exception as e:
        print('IsolationForest failed:', e)

    # salva histogramas simples para features chave incluindo target
    keys = ['Consumo'] + [c for c in ['TEMPERATURA MEDIA, MENSAL (AUT)(°C)', 'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)', 'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)'] if c in df.columns]
    for c in keys:
        if not PLOTTING_AVAILABLE:
            continue
        try:
            plt.figure()
            sns.histplot(df[c].dropna(), kde=True)
            plt.title(f'Histogram {c}')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f'hist_{c.replace("/","_").replace(" ","_")}.png'))
            plt.close()
        except Exception:
            pass

    print('Analysis complete. Files in', OUT_DIR)


if __name__ == '__main__':
    run_analysis()
