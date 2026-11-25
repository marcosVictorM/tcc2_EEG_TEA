# 06_diagnostico_ica.py
"""
Diagnóstico do efeito do ICA:
  1) Quantos componentes oculares o ICA removeu em ASD vs TD (se diferir, o artefato ocular carregava sinal do grupo).
  2) AUC (independe de limiar) do SVM de repouso ANTES vs DEPOIS do ICA.
  3) Features mais correlacionadas com o diagnóstico antes vs depois, destacando as frontais (Fp1, Fp2, F7, F8).

Para a comparação 2 e 3, renomeie os arquivos antigos (sem ICA) antes de reextrair, ex.:
  X_features.npy -> X_features_sem_ica.npy | y_labels.npy -> y_labels_sem_ica.npy | metadata.csv -> metadata_sem_ica.csv
"""
import os
import importlib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import config
from metricas import nomes_features_assinatura

ml = importlib.import_module('02_treinamento_ml')  # reutiliza extrair_assinatura_limpa

OLD_X = os.path.join(config.BASE_DIR, 'X_features_sem_ica.npy')
OLD_Y = os.path.join(config.BASE_DIR, 'y_labels_sem_ica.npy')
OLD_META = os.path.join(config.BASE_DIR, 'metadata_sem_ica.csv')
FRONTAIS = ['Fp1', 'Fp2', 'F7', 'F8']


def assinaturas_rest(X, y, df_meta):
    mask = (df_meta['paradigma'].values == 'rest') if 'paradigma' in df_meta.columns else np.ones(len(y), bool)
    Xr, yr, pids = X[mask], y[mask], df_meta['participante_id'].values[mask]
    A, Y, P = [], [], []
    for pid in pd.unique(pids):
        m = (pids == pid)
        A.append(ml.extrair_assinatura_limpa(Xr[m]))
        Y.append(int(yr[m][0]))
        P.append(pid)
    return np.array(A), np.array(Y), np.array(P)


def auc_svm(A, Y):
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('selector', SelectKBest(f_classif, k=60)),
        ('clf', SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=config.RANDOM_STATE))
    ])
    aucs = []
    for seed in range(5):  # várias partições: AUC com n=103 oscila bastante
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        prob = cross_val_predict(pipe, A, Y, cv=cv, method='predict_proba')[:, 1]
        aucs.append(roc_auc_score(Y, prob))
    return np.mean(aucs), np.std(aucs)


def top_correlacoes(A, Y, nomes, n=15):
    Ac = A - A.mean(0)
    yc = Y - Y.mean()
    den = np.sqrt((Ac ** 2).sum(0) * (yc ** 2).sum())
    r = np.divide(Ac.T @ yc, den, out=np.zeros(A.shape[1]), where=den > 0)
    idx = np.argsort(-np.abs(r))[:n]
    return [(nomes[i], r[i]) for i in idx], r


def eh_frontal(nome):
    tokens = nome.replace('|', '_').replace('-', '_').replace('/', '_').split('_')
    return any(t in FRONTAIS for t in tokens)


def main():
    print("=" * 70)
    print("DIAGNÓSTICO DO EFEITO DO ICA")
    print("=" * 70)

    X = np.load(config.X_FEATURES_PATH)
    y = np.load(config.Y_LABELS_PATH)
    df = pd.read_csv(config.METADATA_PATH)

    # ---- 1) componentes removidos por grupo
    if 'n_ica_removidos' in df.columns:
        u = df.drop_duplicates(['participante_id', 'paradigma'])
        print("\n1) Componentes removidos pelo ICA (por sujeito/paradigma):")
        print(u.groupby(['paradigma', 'grupo'])['n_ica_removidos'].agg(['count', 'mean', 'median']).round(2).to_string())
        for p in u['paradigma'].unique():
            a = u[(u.paradigma == p) & (u.grupo == 'ASD')]['n_ica_removidos']
            t = u[(u.paradigma == p) & (u.grupo == 'TD')]['n_ica_removidos']
            if len(a) > 1 and len(t) > 1:
                pval = mannwhitneyu(a, t).pvalue
                print(f"   [{p}] Mann-Whitney ASD vs TD: p = {pval:.4f}"
                      f"{'  <- grupos diferem em artefato ocular' if pval < 0.05 else ''}")
        print(f"   Sujeitos com 0 componentes removidos: {(u['n_ica_removidos'] == 0).mean():.0%}")
    else:
        print("\n1) metadata.csv não tem 'n_ica_removidos' (extraído sem ICA).")

    # ---- 2) AUC antes/depois, separando perda de sujeitos x efeito do ICA
    A_new, Y_new, P_new = assinaturas_rest(X, y, df)
    nomes = nomes_features_assinatura()

    tem_antigo = all(os.path.exists(p) for p in (OLD_X, OLD_Y, OLD_META))
    if not tem_antigo:
        m, s = auc_svm(A_new, Y_new)
        print(f"\n2) AUC SVM (REST, 5x5-fold) DEPOIS do ICA: {m:.3f} +- {s:.3f}  (n={len(Y_new)})")
        print("   (arquivos *_sem_ica não encontrados: comparação antes/depois ignorada)")
        return

    X0, y0, df0 = np.load(OLD_X), np.load(OLD_Y), pd.read_csv(OLD_META)
    A_old, Y_old, P_old = assinaturas_rest(X0, y0, df0)

    idx_old = {p: i for i, p in enumerate(P_old)}
    idx_new = {p: i for i, p in enumerate(P_new)}
    comuns = [p for p in P_new if p in idx_old]
    perdidos = [p for p in P_old if p not in idx_new]
    io = [idx_old[p] for p in comuns]
    inw = [idx_new[p] for p in comuns]
    A_old_c, Y_old_c = A_old[io], Y_old[io]
    A_new_c, Y_new_c = A_new[inw], Y_new[inw]

    print(f"\n2) Sujeitos: antes={len(P_old)} | depois={len(P_new)} | em comum={len(comuns)}")
    print(f"   Perdidos após o ICA ({len(perdidos)}): {perdidos}")
    perd_y = [int(Y_old[idx_old[p]]) for p in perdidos]
    print(f"   Perdidos por grupo: TD={perd_y.count(0)} | ASD={perd_y.count(1)}")

    m_a, s_a = auc_svm(A_old, Y_old)
    m_b, s_b = auc_svm(A_old_c, Y_old_c)
    m_c, s_c = auc_svm(A_new_c, Y_new_c)
    print("\n   AUC SVM (REST, 5x5-fold):")
    print(f"   (A) ANTES do ICA, todos os sujeitos antigos (n={len(Y_old)}):   {m_a:.3f} +- {s_a:.3f}")
    print(f"   (B) ANTES do ICA, só os sujeitos em comum   (n={len(Y_old_c)}):   {m_b:.3f} +- {s_b:.3f}")
    print(f"   (C) DEPOIS do ICA, só os sujeitos em comum  (n={len(Y_new_c)}):   {m_c:.3f} +- {s_c:.3f}")
    print("   -> A vs B: efeito de PERDER sujeitos.  B vs C: efeito REAL do ICA (mesmos sujeitos).")
    print("   -> Se B >> C, o artefato ocular carregava informação do grupo.")

    A_old, Y_old, A_new, Y_new = A_old_c, Y_old_c, A_new_c, Y_new_c

    # ---- 3) top correlações antes vs depois
    for rotulo, A, Y in (("ANTES", A_old, Y_old), ("DEPOIS", A_new, Y_new)):
        top, r = top_correlacoes(A, Y, nomes)
        n_front = sum(eh_frontal(nm) for nm, _ in top)
        print(f"\n3) Top 15 |r| com o diagnóstico {rotulo} do ICA  (frontais Fp1/Fp2/F7/F8: {n_front}/15)")
        for nm, v in top[:8]:
            print(f"   r={v:+.3f}  {nm}{'  [frontal]' if eh_frontal(nm) else ''}")
        print(f"   maior |r| = {np.max(np.abs(r)):.3f}")


if __name__ == '__main__':
    main()
