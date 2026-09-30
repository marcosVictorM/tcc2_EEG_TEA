# 02_treinamento_ml.py
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import config

def extrair_assinatura_limpa(X_bloco):
    """Extrai Conectividade Z + PSD Relativo + Razões Espectrais de um paradigma para 1 sujeito"""
    n_canais = 19 if config.USAR_SISTEMA_10_20 else 64
    n_conn = (n_canais * (n_canais - 1)) // 2
    n_cols_banda = n_conn + n_canais

    conn_bandas = []
    psd_bandas = []

    for b in range(4):
        offset = b * n_cols_banda
        conn_b = np.clip(X_bloco[:, offset : offset + n_conn], -0.999, 0.999)
        conn_bandas.append(np.arctanh(conn_b))
        
        psd_b = np.maximum(X_bloco[:, offset + n_conn : offset + n_cols_banda], 1e-12)
        psd_bandas.append(psd_b)

    psd_soma_total = psd_bandas[0] + psd_bandas[1] + psd_bandas[2] + psd_bandas[3]
    psd_rel_bandas = [psd_b / psd_soma_total for psd_b in psd_bandas]

    ratio_theta_beta  = np.log10(psd_bandas[1] / psd_bandas[3])
    ratio_theta_alpha = np.log10(psd_bandas[1] / psd_bandas[2])
    ratio_alpha_beta  = np.log10(psd_bandas[2] / psd_bandas[3])
    ratio_delta_theta = np.log10(psd_bandas[0] / psd_bandas[1])

    epocas_enriq = np.hstack(
        conn_bandas + psd_rel_bandas + [ratio_theta_beta, ratio_theta_alpha, ratio_alpha_beta, ratio_delta_theta]
    )
    epocas_enriq = np.nan_to_num(epocas_enriq, nan=0.0, posinf=0.0, neginf=0.0)

    energia_epocas = np.mean(psd_soma_total, axis=1)
    q25, q75 = np.percentile(energia_epocas, [25, 75])
    iqr = q75 - q25
    mask_limpa = (energia_epocas >= (q25 - 1.5 * iqr)) & (energia_epocas <= (q75 + 1.5 * iqr))
    epocas_limpas = epocas_enriq[mask_limpa] if np.sum(mask_limpa) >= 10 else epocas_enriq

    mediana_suj = np.median(epocas_limpas, axis=0)
    iqr_suj = np.percentile(epocas_limpas, 75, axis=0) - np.percentile(epocas_limpas, 25, axis=0)
    return np.concatenate([mediana_suj, iqr_suj])

def avaliar_pipeline_calibrado(X_mat, y_vec, pipe, nome_modelo):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X_mat, y_vec):
        X_tr, y_tr = X_mat[train_idx], y_vec[train_idx]
        X_te, y_te = X_mat[test_idx], y_vec[test_idx]

        pipe.fit(X_tr, y_tr)
        prob_tr = pipe.predict_proba(X_tr)[:, 1]
        
        melhor_th = 0.50
        maior_j = -1.0
        for th in np.linspace(0.30, 0.70, 41):
            j_score = balanced_accuracy_score(y_tr, (prob_tr >= th).astype(int))
            if j_score > maior_j:
                maior_j = j_score
                melhor_th = th

        prob_te = pipe.predict_proba(X_te)[:, 1]
        preds_te = (prob_te >= melhor_th).astype(int)

        y_true_all.extend(y_te)
        y_pred_all.extend(preds_te)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc_bruta = np.mean(y_true_all == y_pred_all)
    acc_bal = balanced_accuracy_score(y_true_all, y_pred_all)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)

    print("=======================================================")
    print(f"{nome_modelo}")
    print("=======================================================")
    print(f"Diagnósticos Corretos:        {tn+tp}/{len(y_vec)} ({acc_bruta:.2%})")
    print(f"Acurácia Clínica Balanceada:  {acc_bal:.2%}")
    print(f"Acerto em Neurotípicos (TD):  {tn}/{tn+fp} ({spec:.2%})")
    print(f"Acerto em Autismo      (ASD): {tp}/{tp+fn} ({sens:.2%})")
    print("=======================================================\n")

def executar_consolidacao_fase1():
    print("Iniciando Consolidação Oficial da Fase 1 (Machine Learning Clássico - SFARI BIDS)...\n")

    if not os.path.exists(config.X_FEATURES_PATH):
        print("Erro: Ficheiros .npy não encontrados.")
        return

    X = np.load(config.X_FEATURES_PATH)
    y = np.load(config.Y_LABELS_PATH)
    df_meta = pd.read_csv(config.METADATA_PATH)

    if 'paradigma' not in df_meta.columns:
        df_meta['paradigma'] = 'rest'

    # 1. Matriz de Repouso Isolado (103 pacientes x 1672 biomarcadores)
    mask_rest = (df_meta['paradigma'].values == 'rest')
    X_rest_ep = X[mask_rest]
    y_rest_ep = y[mask_rest]
    pids_rest_ep = df_meta['participante_id'].values[mask_rest]

    pids_base = pd.unique(pids_rest_ep)
    X_rest_suj, y_rest_suj = [], []
    dict_rest = {}

    for pid in pids_base:
        m_s = (pids_rest_ep == pid)
        assinatura = extrair_assinatura_limpa(X_rest_ep[m_s])
        X_rest_suj.append(assinatura)
        y_rest_suj.append(int(y_rest_ep[m_s][0]))
        dict_rest[pid] = assinatura

    X_rest_suj = np.array(X_rest_suj)
    y_rest_suj = np.array(y_rest_suj)

    pipe_svm_rest = Pipeline([
        ('scaler', RobustScaler()),
        ('selector', SelectKBest(f_classif, k=60)),
        ('clf', SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=config.RANDOM_STATE))
    ])

    avaliar_pipeline_calibrado(
        X_rest_suj, y_rest_suj, pipe_svm_rest,
        "MODELO CAMPEÃO EM EQUILÍBRIO CLÍNICO: SVM RBF (Repouso Isolado - 1.672 feats)"
    )

    # 2. Matriz Multi-Paradigma (103 pacientes x 8360 biomarcadores)
    paradigmas_presentes = [p for p in ['rest', 'fast', 'assr'] if p in df_meta['paradigma'].unique()]
    if len(paradigmas_presentes) > 1:
        n_feats_unit = X_rest_suj.shape[1]
        dict_por_p = {'rest': dict_rest}
        for p in ['fast', 'assr']:
            if p in paradigmas_presentes:
                dict_por_p[p] = {}
                m_p = (df_meta['paradigma'].values == p)
                X_p, pids_p = X[m_p], df_meta['participante_id'].values[m_p]
                for pid in pd.unique(pids_p):
                    dict_por_p[p][pid] = extrair_assinatura_limpa(X_p[pids_p == pid])

        X_multi_suj = []
        for pid in pids_base:
            v_rest = dict_por_p['rest'][pid]
            blocos = [v_rest]
            for p in ['fast', 'assr']:
                if p in dict_por_p:
                    v_p = dict_por_p[p].get(pid, np.full(n_feats_unit, np.nan))
                    blocos.append(v_p)
                    blocos.append(v_p - v_rest)
            X_multi_suj.append(np.concatenate(blocos))
        X_multi_suj = np.array(X_multi_suj)

        pipe_ensemble_multi = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=80)),
            ('clf', VotingClassifier(
                estimators=[
                    ('svm', SVC(C=1.2, kernel='rbf', class_weight='balanced', probability=True, random_state=config.RANDOM_STATE)),
                    ('et', ExtraTreesClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=config.RANDOM_STATE)),
                    ('lr', LogisticRegression(C=0.3, class_weight='balanced', random_state=config.RANDOM_STATE))
                ],
                voting='soft',
                weights=[2, 1, 1]
            ))
        ])

        avaliar_pipeline_calibrado(
            X_multi_suj, y_rest_suj, pipe_ensemble_multi,
            "MODELO CAMPEÃO EM SENSIBILIDADE / TRIAGEM: Ensemble Híbrido Multi-Paradigma (8.360 feats)"
        )

if __name__ == '__main__':
    executar_consolidacao_fase1()