# metricas.py
"""
Funções compartilhadas pelos scripts 02, 03 e 05:
  - relatorio_metricas(): acurácia (score), precisão, recall, F1 (F-measure) por classe e médias
  - analisar_correlacao_target(): correlação de Pearson de cada feature com o diagnóstico (TD=0, ASD=1),
    lista as features com |r| >= 0.7 e salva CSV + heatmap da matriz de correlação.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # só salva PNG, não abre janela
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import config

BANDAS = ['delta', 'theta', 'alpha', 'beta']


# ----------------------------------------------------------------------
# 1) MÉTRICAS DE CLASSIFICAÇÃO
# ----------------------------------------------------------------------
def relatorio_metricas(y_true, y_pred, titulo=""):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    m = {
        'accuracy (score)':    accuracy_score(y_true, y_pred),
        'balanced_accuracy':   balanced_accuracy_score(y_true, y_pred),
        'precision_ASD':       precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_ASD':          recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1_ASD':              f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'precision_TD':        precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'recall_TD':           recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'f1_TD':               f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        'precision_macro':     precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro':        recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro':            f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted':         f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    print("-------------------------------------------------------")
    print(f"MÉTRICAS DETALHADAS {('- ' + titulo) if titulo else ''}")
    print("-------------------------------------------------------")
    print(classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=['Neurotípico (TD)', 'Autismo (ASD)'],
        digits=4, zero_division=0
    ))
    print(f"Accuracy (score):     {m['accuracy (score)']:.4f}")
    print(f"Balanced accuracy:    {m['balanced_accuracy']:.4f}")
    print(f"Precision (ASD):      {m['precision_ASD']:.4f}")
    print(f"Recall    (ASD):      {m['recall_ASD']:.4f}   <- sensibilidade")
    print(f"F1-score  (ASD):      {m['f1_ASD']:.4f}")
    print(f"F1-score  (macro):    {m['f1_macro']:.4f}")
    print(f"F1-score  (weighted): {m['f1_weighted']:.4f}")
    print("-------------------------------------------------------\n")
    return m


# ----------------------------------------------------------------------
# 2) NOMES DAS FEATURES (para saber QUAL feature é correlacionada)
# ----------------------------------------------------------------------
def _canais():
    if config.USAR_SISTEMA_10_20:
        return list(config.CANAIS_10_20)
    return [f'ch{i}' for i in range(64)]


def nomes_features_assinatura():
    """
    Nomes das 1672 features de 1 paradigma (mesma ordem de extrair_assinatura_limpa):
    [mediana | iqr] x [conn(4 bandas), psd_rel(4 bandas), 4 razões espectrais por canal]
    """
    canais = _canais()
    n = len(canais)
    iu, ju = np.triu_indices(n, k=1)
    pares = [f"{canais[i]}-{canais[j]}" for i, j in zip(iu, ju)]

    base = []
    for b in BANDAS:
        base += [f"conn_{b}_{p}" for p in pares]
    for b in BANDAS:
        base += [f"psdrel_{b}_{c}" for c in canais]
    for r in ['theta/beta', 'theta/alpha', 'alpha/beta', 'delta/theta']:
        base += [f"ratio_{r}_{c}" for c in canais]

    return [f"mediana|{x}" for x in base] + [f"iqr|{x}" for x in base]


def nomes_features_multi():
    """Nomes das 8360 features: rest, fast, fast-rest, assr, assr-rest (5 x 1672)."""
    base = nomes_features_assinatura()
    nomes = [f"rest|{x}" for x in base]
    for p in ['fast', 'assr']:
        nomes += [f"{p}|{x}" for x in base]
        nomes += [f"{p}-rest|{x}" for x in base]
    return nomes


# ----------------------------------------------------------------------
# 3) CORRELAÇÃO COM O TARGET
# ----------------------------------------------------------------------
def analisar_correlacao_target(X, y, nomes=None, limiar=0.7, top_n=20,
                               prefixo='correlacao', titulo=""):
    """
    Correlação de Pearson (ponto-bisserial) entre cada feature e o diagnóstico (TD=0, ASD=1).
      r > 0  -> feature MAIOR em ASD
      r < 0  -> feature MAIOR em TD
    Mostra quantas features têm |r| >= limiar (0.7), lista as top_n e salva:
      <prefixo>_ranking.csv  e  <prefixo>_matriz.png (matriz de correlação das top_n + target)
    ATENÇÃO: é análise exploratória no conjunto todo; NÃO use para selecionar features
    antes da validação cruzada (vazamento de dados).
    """
    X = np.array(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # NaN (ex.: sujeito sem FAST/ASSR) -> mediana da coluna
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        idx = np.where(np.isnan(X))
        X[idx] = med[idx[1]]

    if nomes is None or len(nomes) != X.shape[1]:
        nomes = [f"feat_{i}" for i in range(X.shape[1])]

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    r = np.divide(Xc.T @ yc, den, out=np.zeros(X.shape[1]), where=den > 0)

    df = pd.DataFrame({'feature': nomes, 'r': r, 'abs_r': np.abs(r)})
    df = df.sort_values('abs_r', ascending=False).reset_index(drop=True)

    n_fortes = int((df['abs_r'] >= limiar).sum())
    n_pos = int((df['r'] >= limiar).sum())
    n_neg = int((df['r'] <= -limiar).sum())

    print("-------------------------------------------------------")
    print(f"CORRELAÇÃO COM O TARGET {('- ' + titulo) if titulo else ''}")
    print("-------------------------------------------------------")
    print(f"Features analisadas:                 {X.shape[1]}  (amostras: {X.shape[0]})")
    print(f"Features com r >=  {limiar}:  {n_pos}")
    print(f"Features com r <= -{limiar}:  {n_neg}")
    print(f"Total com |r| >= {limiar}:           {n_fortes}")
    if n_fortes == 0:
        print(f"(Nenhuma feature atinge |r| >= {limiar}. Maior |r| observado: {df['abs_r'].iloc[0]:.3f})")
    print(f"\nTop {top_n} features mais correlacionadas com o diagnóstico:")
    for i in range(min(top_n, len(df))):
        sinal = "maior em ASD" if df['r'].iloc[i] > 0 else "maior em TD "
        marca = " <==" if df['abs_r'].iloc[i] >= limiar else ""
        print(f"  {i+1:>2}. r={df['r'].iloc[i]:+.3f} ({sinal}) {df['feature'].iloc[i]}{marca}")

    # --- salva ranking completo
    pasta = config.BASE_DIR
    caminho_csv = os.path.join(pasta, f"{prefixo}_ranking.csv")
    df.to_csv(caminho_csv, index=False)

    # --- matriz de correlação entre as top_n features + target
    top_idx = [nomes.index(f) for f in df['feature'].iloc[:top_n]]
    M = np.column_stack([X[:, top_idx], y])
    corr = np.corrcoef(M, rowvar=False)
    corr = np.nan_to_num(corr)
    rotulos = [f.replace('mediana|', 'med|') for f in df['feature'].iloc[:top_n]] + ['TARGET (ASD)']

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label='Correlação de Pearson')
    ax.set_xticks(range(len(rotulos)))
    ax.set_yticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, rotation=90, fontsize=7)
    ax.set_yticklabels(rotulos, fontsize=7)
    ax.set_title(f"Matriz de correlação: top {top_n} features + target {titulo}", fontsize=11)
    fig.tight_layout()
    caminho_png = os.path.join(pasta, f"{prefixo}_matriz.png")
    plt.savefig(caminho_png, dpi=200)
    plt.close(fig)

    print(f"\nRanking completo salvo em: {caminho_csv}")
    print(f"Matriz de correlação salva em: {caminho_png}")
    print("-------------------------------------------------------\n")
    return df
