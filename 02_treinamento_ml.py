# 02_treinamento_ml.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import config

def construir_features_relativas_e_limpar_epocas(X, y, participant_ids):
    """
    1. Converte PSD absoluto em PSD Relativo (%) + Razões Espectrais (Theta/Beta, Theta/Alpha)
    2. Aplica Transformada Z de Fisher (arctanh) na Conectividade de Pearson
    3. Remove épocas com artefactos severos (outliers de energia intra-sujeito)
    4. Agrega o perfil limpo por sujeito (Mediana e Desvio Padrão Robusto)
    """
    n_canais = 19 if config.USAR_SISTEMA_10_20 else 64
    n_conn = (n_canais * (n_canais - 1)) // 2  # 171 para 19 canais
    n_cols_banda = n_conn + n_canais           # 190 para 19 canais

    conn_bandas = []
    psd_bandas = []

    for b in range(4):
        offset = b * n_cols_banda
        conn_b = np.clip(X[:, offset : offset + n_conn], -0.999, 0.999)
        # Transformada Z de Fisher estabiliza a variância da correlação de Pearson
        conn_bandas.append(np.arctanh(conn_b))
        
        psd_b = np.maximum(X[:, offset + n_conn : offset + n_cols_banda], 1e-12)
        psd_bandas.append(psd_b)

    # Soma da potência nas 4 bandas para cada canal (N_epocas, 19)
    psd_soma_total = psd_bandas[0] + psd_bandas[1] + psd_bandas[2] + psd_bandas[3]

    # Potência Relativa por banda (elimina diferenças de crânio/impedância entre pacientes)
    psd_rel_bandas = [psd_b / psd_soma_total for psd_b in psd_bandas]

    # Biomarcadores clássicos de neurodesenvolvimento (Razões Espectrais em log)
    ratio_theta_beta  = np.log10(psd_bandas[1] / psd_bandas[3])
    ratio_theta_alpha = np.log10(psd_bandas[1] / psd_bandas[2])
    ratio_alpha_beta  = np.log10(psd_bandas[2] / psd_bandas[3])
    ratio_delta_theta = np.log10(psd_bandas[0] / psd_bandas[1])

    # Matriz enriquecida por época
    X_epocas_enriquecido = np.hstack(
        conn_bandas + psd_rel_bandas + [ratio_theta_beta, ratio_theta_alpha, ratio_alpha_beta, ratio_delta_theta]
    )
    X_epocas_enriquecido = np.nan_to_num(X_epocas_enriquecido, nan=0.0, posinf=0.0, neginf=0.0)

    # Agregação por Paciente com rejeição de épocas contaminadas por artefactos
    pids_unicos = []
    X_sujeitos = []
    y_sujeitos = []

    for pid in pd.unique(participant_ids):
        mask = (participant_ids == pid)
        epocas_pid = X_epocas_enriquecido[mask]
        energia_epocas = np.mean(psd_soma_total[mask], axis=1)

        # Rejeita épocas com picos de movimento/piscadas (acima de Q75 + 1.5*IQR ou abaixo de Q25 - 1.5*IQR)
        q25, q75 = np.percentile(energia_epocas, [25, 75])
        iqr = q75 - q25
        mask_limpa = (energia_epocas >= (q25 - 1.5 * iqr)) & (energia_epocas <= (q75 + 1.5 * iqr))
        if np.sum(mask_limpa) >= 10:
            epocas_limpas = epocas_pid[mask_limpa]
        else:
            epocas_limpas = epocas_pid

        # Extrai a assinatura central (mediana) e a variabilidade temporal (IQR) do paciente
        mediana_suj = np.median(epocas_limpas, axis=0)
        iqr_suj = np.percentile(epocas_limpas, 75, axis=0) - np.percentile(epocas_limpas, 25, axis=0)
        
        X_sujeitos.append(np.concatenate([mediana_suj, iqr_suj]))
        y_sujeitos.append(int(y[mask][0]))
        pids_unicos.append(pid)

    return np.array(X_sujeitos), np.array(y_sujeitos), np.array(pids_unicos)

def avaliar_modelos():
    print("Iniciando Fase 1.5: Engenharia Espectral Relativa + Benchmark ML Clássico...")

    if not os.path.exists(config.X_FEATURES_PATH):
        print("Erro: Ficheiros .npy não encontrados.")
        return

    X = np.load(config.X_FEATURES_PATH)
    y = np.load(config.Y_LABELS_PATH)
    df_meta = pd.read_csv(config.METADATA_PATH)
    participant_ids = df_meta['participante_id'].values

    X_suj, y_suj, pids = construir_features_relativas_e_limpar_epocas(X, y, participant_ids)
    print(f"Assinaturas limpas geradas: {X_suj.shape[0]} pacientes × {X_suj.shape[1]} biomarcadores (Conectividade Z + PSD Relativo + Razões).\n")

    modelos = {
        "SVM (Kernel RBF Balanceado)": Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=60)),
            ('clf', SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=config.RANDOM_STATE))
        ]),
        "Random Forest (Balanceado)": Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=80)),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight='balanced_subsample', random_state=config.RANDOM_STATE, n_jobs=-1))
        ]),
        "Extra Trees (Balanceado)": Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif, k=80)),
            ('clf', ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, class_weight='balanced', random_state=config.RANDOM_STATE, n_jobs=-1))
        ]),
        "Regressão Logística (Lasso L1)": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=60)),
            ('clf', LogisticRegression(penalty='l1', solver='liblinear', C=0.25, class_weight='balanced', random_state=config.RANDOM_STATE))
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
            
            # Calibra o limiar ótimo no próprio conjunto de treino (Índice de Youden)
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
        print(f"   Acurácia Bruta: {acc_bruta:.2%} ({tn+tp}/103) | Balanceada: {acc_bal:.2%}")
        print(f"   Especificidade (TD): {tn}/40 ({spec:.2%}) | Sensibilidade (ASD): {tp}/63 ({sens:.2%})\n")

        if acc_bal > melhor_bal_acc:
            melhor_bal_acc = acc_bal
            melhor_nome = nome
            melhor_resumo = (acc_bruta, acc_bal, tn, fp, fn, tp, spec, sens)

    acc_bruta, acc_bal, tn, fp, fn, tp, spec, sens = melhor_resumo
    print("=======================================================")
    print(f"MELHOR MODELO CLÁSSICO: {melhor_nome}")
    print("=======================================================")
    print(f"Diagnósticos Corretos:        {tn+tp}/103 ({acc_bruta:.2%})")
    print(f"Acurácia Clínica Balanceada:  {acc_bal:.2%}")
    print(f"Acerto em Neurotípicos (TD):  {tn}/40 ({spec:.2%})")
    print(f"Acerto em Autismo      (ASD): {tp}/63 ({sens:.2%})")
    print("=======================================================\n")

if __name__ == '__main__':
    avaliar_modelos()