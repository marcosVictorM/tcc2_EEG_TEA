# 02_treinamento_ml.py
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
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

def construir_matriz_multiparadigma(X, y, df_meta):
    if 'paradigma' not in df_meta.columns:
        df_meta['paradigma'] = 'rest'

    paradigmas_presentes = [p for p in ['rest', 'fast', 'assr'] if p in df_meta['paradigma'].unique()]
    print(f"Paradigmas detectados na matriz: {paradigmas_presentes}")

    if 'rest' in paradigmas_presentes:
        pids_base = df_meta[df_meta['paradigma'] == 'rest']['participante_id'].unique()
    else:
        pids_base = df_meta['participante_id'].unique()

    n_feats_unit = 1672 if config.USAR_SISTEMA_10_20 else 16896
    dict_por_paradigma = {p: {} for p in paradigmas_presentes}
    labels_por_pid = {}

    for p in paradigmas_presentes:
        mask_p = (df_meta['paradigma'].values == p)
        X_p = X[mask_p]
        y_p = y[mask_p]
        pids_p = df_meta['participante_id'].values[mask_p]

        for pid in pd.unique(pids_p):
            mask_suj = (pids_p == pid)
            dict_por_paradigma[p][pid] = extrair_assinatura_limpa(X_p[mask_suj])
            labels_por_pid[pid] = int(y_p[mask_suj][0])

    X_fused_list = []
    y_fused_list = []

    for pid in pids_base:
        blocos_sujeito = []
        v_rest = dict_por_paradigma.get('rest', {}).get(pid, np.full(n_feats_unit, np.nan))

        for p in paradigmas_presentes:
            v_p = dict_por_paradigma[p].get(pid, np.full(n_feats_unit, np.nan))
            blocos_sujeito.append(v_p)
            # Calcula o vetor de Reatividade Dinâmica (Delta = Tarefa Sensorial - Repouso)
            if p != 'rest' and 'rest' in paradigmas_presentes:
                delta_p = v_p - v_rest
                blocos_sujeito.append(delta_p)

        X_fused_list.append(np.concatenate(blocos_sujeito))
        y_fused_list.append(labels_por_pid[pid])

    return np.array(X_fused_list), np.array(y_fused_list), pids_base

def avaliar_fusao_multiparadigma():
    print("Iniciando Fase 1.6: Fusão Multi-Paradigma (Repouso + Estímulo Sensorial + Reatividade)...")

    if not os.path.exists(config.X_FEATURES_PATH):
        print("Erro: Ficheiros .npy não encontrados. Rode 'python 01_extracao.py' primeiro.")
        return

    X = np.load(config.X_FEATURES_PATH)
    y = np.load(config.Y_LABELS_PATH)
    df_meta = pd.read_csv(config.METADATA_PATH)

    X_suj, y_suj, pids = construir_matriz_multiparadigma(X, y, df_meta)
    print(f"Matriz Multi-Paradigma construída: {X_suj.shape[0]} pacientes × {X_suj.shape[1]} biomarcadores integrados.\n")

    modelos = {
        "SVM Multi-Paradigma (RBF k=70)": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=min(70, X_suj.shape[1]))),
            ('clf', SVC(C=1.2, kernel='rbf', class_weight='balanced', probability=True, random_state=config.RANDOM_STATE))
        ]),
        "SVM Multi-Paradigma (RBF k=100)": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=min(100, X_suj.shape[1]))),
            ('clf', SVC(C=1.5, kernel='rbf', class_weight='balanced', probability=True, random_state=config.RANDOM_STATE))
        ]),
        "Extra Trees Multi-Paradigma": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=min(90, X_suj.shape[1]))),
            ('clf', ExtraTreesClassifier(n_estimators=400, max_depth=6, min_samples_leaf=2, class_weight='balanced', random_state=config.RANDOM_STATE, n_jobs=-1))
        ]),
        "Ensemble Híbrido (SVM + ExtraTrees + LogReg)": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=min(80, X_suj.shape[1]))),
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
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    melhor_bal_acc = 0.0
    melhor_nome = ""
    melhor_resumo = None

    for nome, pipe in modelos.items():
        y_true_all, y_pred_all = [], []
        
        for train_idx, test_idx in cv.split(X_suj, y_suj):
            X_tr, y_tr = X_suj[train_idx], y_suj[train_idx]
            X_te, y_te = X_suj[test_idx], y_suj[test_idx]

            pipe.fit(X_tr, y_tr)
            
            prob_tr = pipe.predict_proba(X_tr)[:, 1]
            limiares = np.linspace(0.30, 0.70, 41)
            melhor_th = 0.50
            maior_j = -1.0
            for th in limiares:
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

        print(f"-> {nome}:")
        print(f"   Acurácia Bruta: {acc_bruta:.2%} ({tn+tp}/{len(y_suj)}) | Balanceada: {acc_bal:.2%}")
        print(f"   Especificidade (TD): {tn}/{tn+fp} ({spec:.2%}) | Sensibilidade (ASD): {tp}/{tp+fn} ({sens:.2%})\n")

        if acc_bal > melhor_bal_acc:
            melhor_bal_acc = acc_bal
            melhor_nome = nome
            melhor_resumo = (acc_bruta, acc_bal, tn, fp, fn, tp, spec, sens)

    acc_bruta, acc_bal, tn, fp, fn, tp, spec, sens = melhor_resumo
    print("=======================================================")
    print(f"MELHOR MODELO MULTI-PARADIGMA: {melhor_nome}")
    print("=======================================================")
    print(f"Diagnósticos Corretos:        {tn+tp}/{len(y_suj)} ({acc_bruta:.2%})")
    print(f"Acurácia Clínica Balanceada:  {acc_bal:.2%}")
    print(f"Acerto em Neurotípicos (TD):  {tn}/{tn+fp} ({spec:.2%})")
    print(f"Acerto em Autismo      (ASD): {tp}/{tp+fn} ({sens:.2%})")
    print("=======================================================\n")

if __name__ == '__main__':
    avaliar_fusao_multiparadigma()