# 03_treinamento_dl.py
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import config

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

def fixar_sementes(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def reconstruir_tensor_topologico_e_espectral(X_epocas, participant_ids, y_epocas):
    """
    Reconstrói a topologia espacial 3D do cérebro a partir das features extraídas:
    1. Tensor de Conectividade Cortical 3D: (N_sujeitos, 19 canais, 19 canais, 4 bandas)
       Preserva qual eletrodo está conectado com qual nas bandas Delta, Theta, Alpha e Beta.
    2. Vetor de Biomarcadores Espectrais Limpos (PSD Relativo + Razões + IQR).
    """
    n_canais = 19 if config.USAR_SISTEMA_10_20 else 64
    n_conn = (n_canais * (n_canais - 1)) // 2
    n_cols_banda = n_conn + n_canais
    upper_idx = np.triu_indices(n_canais, k=1)

    pids_unicos = pd.unique(participant_ids)
    tensores_3d = []
    vetores_espectrais = []
    y_sujeitos = []

    for pid in pids_unicos:
        mask = (participant_ids == pid)
        X_bloco = X_epocas[mask]

        conn_bandas = []
        psd_bandas = []

        for b in range(4):
            offset = b * n_cols_banda
            conn_b = np.clip(X_bloco[:, offset : offset + n_conn], -0.999, 0.999)
            conn_bandas.append(np.arctanh(conn_b))

            psd_b = np.maximum(X_bloco[:, offset + n_conn : offset + n_cols_banda], 1e-12)
            psd_bandas.append(psd_b)

        psd_soma_total = psd_bandas[0] + psd_bandas[1] + psd_bandas[2] + psd_bandas[3]
        energia_epocas = np.mean(psd_soma_total, axis=1)

        # Limpeza de épocas com artefactos oculares/musculares (filtro IQR intra-sujeito)
        q25, q75 = np.percentile(energia_epocas, [25, 75])
        iqr = q75 - q25
        mask_limpa = (energia_epocas >= (q25 - 1.5 * iqr)) & (energia_epocas <= (q75 + 1.5 * iqr))
        if np.sum(mask_limpa) < 10:
            mask_limpa = np.ones(len(energia_epocas), dtype=bool)

        # 1. Constrói a Matriz de Adjacência Funcional 3D (19 x 19 x 4) das épocas limpas
        matriz_suj_3d = np.zeros((n_canais, n_canais, 4), dtype=np.float32)
        for b in range(4):
            conn_mediana = np.median(conn_bandas[b][mask_limpa], axis=0)
            mat_b = np.zeros((n_canais, n_canais), dtype=np.float32)
            mat_b[upper_idx] = conn_mediana
            mat_b = mat_b + mat_b.T  # Matriz simétrica de conectividade cerebral
            # Insere a potência relativa mediana da banda na diagonal principal
            psd_rel_b = np.median((psd_bandas[b] / psd_soma_total)[mask_limpa], axis=0)
            np.fill_diagonal(mat_b, psd_rel_b)
            matriz_suj_3d[:, :, b] = mat_b

        tensores_3d.append(matriz_suj_3d)

        # 2. Constrói o ramo espectral complementar (PSD Relativo + Razões + Conectividade Z)
        psd_rel_bandas = [psd_b / psd_soma_total for psd_b in psd_bandas]
        ratio_tb = np.log10(psd_bandas[1] / psd_bandas[3])
        ratio_ta = np.log10(psd_bandas[1] / psd_bandas[2])
        ratio_ab = np.log10(psd_bandas[2] / psd_bandas[3])
        ratio_dt = np.log10(psd_bandas[0] / psd_bandas[1])

        epocas_enriq = np.hstack(conn_bandas + psd_rel_bandas + [ratio_tb, ratio_ta, ratio_ab, ratio_dt])
        epocas_enriq = np.nan_to_num(epocas_enriq, nan=0.0, posinf=0.0, neginf=0.0)
        ep_limpas = epocas_enriq[mask_limpa]

        mediana_suj = np.median(ep_limpas, axis=0)
        iqr_suj = np.percentile(ep_limpas, 75, axis=0) - np.percentile(ep_limpas, 25, axis=0)
        vetores_espectrais.append(np.concatenate([mediana_suj, iqr_suj]))

        y_sujeitos.append(int(y_epocas[mask][0]))

    return np.array(tensores_3d, dtype=np.float32), np.array(vetores_espectrais, dtype=np.float32), np.array(y_sujeitos)

def criar_rede_convolucional_hibrida(n_canais, n_feats_selecionadas):
    """
    Arquitetura Deep Learning inspirada na BrainNetCNN + EEGNet:
    - Ramo 1 (Topológico 2D/Espacial): Convoluções 1xN e Nx1 que varrem as conexões de cada eletrodo
      ao longo das 4 bandas de frequência (Delta, Theta, Alpha, Beta).
    - Ramo 2 (Espectral Denso com Residual/Dropout): Processa os biomarcadores mais discriminativos.
    """
    # Ramo 1: Entrada Topológica 3D (19 eletrodos x 19 eletrodos x 4 bandas)
    in_topo = layers.Input(shape=(n_canais, n_canais, 4), name="input_topologico")
    x1 = layers.BatchNormalization()(in_topo)
    # Filtro Edge-to-Node: sintetiza todas as conexões que chegam a cada um dos 19 eletrodos
    x1 = layers.Conv2D(16, kernel_size=(1, n_canais), activation='elu',
                       kernel_regularizer=regularizers.l2(1e-2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.35)(x1)
    # Filtro Node-to-Graph: integra a resposta global dos 19 eletrodos
    x1 = layers.Conv2D(32, kernel_size=(n_canais, 1), activation='elu',
                       kernel_regularizer=regularizers.l2(1e-2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dropout(0.40)(x1)

    # Ramo 2: Entrada de Biomarcadores Espectrais Selecionados
    in_spec = layers.Input(shape=(n_feats_selecionadas,), name="input_espectral")
    x2 = layers.Dense(32, activation='elu', kernel_regularizer=regularizers.l2(1e-2))(in_spec)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.35)(x2)

    # Fusão Neural Profunda
    concat = layers.Concatenate()([x1, x2])
    z = layers.Dense(24, activation='elu', kernel_regularizer=regularizers.l2(1e-2))(concat)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.30)(z)
    out = layers.Dense(1, activation='sigmoid', name="diagnostico_asd")(z)

    model = models.Model(inputs=[in_topo, in_spec], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def executar_treinamento_dl():
    print("Iniciando Fase 2: Deep Learning Topológico (BrainNetCNN Híbrida - SFARI BIDS)...\n")
    fixar_sementes(config.RANDOM_STATE)

    if not os.path.exists(config.X_FEATURES_PATH):
        print("Erro: Ficheiros .npy não encontrados. Rode 'python 01_extracao.py' primeiro.")
        return

    X = np.load(config.X_FEATURES_PATH)
    y = np.load(config.Y_LABELS_PATH)
    df_meta = pd.read_csv(config.METADATA_PATH)

    if 'paradigma' in df_meta.columns:
        mask_rest = (df_meta['paradigma'].values == 'rest')
        X_rest = X[mask_rest]
        y_rest = y[mask_rest]
        pids_rest = df_meta['participante_id'].values[mask_rest]
    else:
        X_rest, y_rest, pids_rest = X, y, df_meta['participante_id'].values

    X_topo, X_spec, y_suj = reconstruir_tensor_topologico_e_espectral(X_rest, pids_rest, y_rest)
    n_canais = X_topo.shape[1]
    k_feats = 60

    print(f"Tensores reconstruídos para {len(y_suj)} pacientes:")
    print(f" -> Tensor Topológico 3D: {X_topo.shape} (Pacientes, Eletrodos, Eletrodos, Bandas)")
    print(f" -> Matriz Espectral:     {X_spec.shape} (Pacientes, Biomarcadores)\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    y_true_all, y_pred_all = [], []

    # Pesos de classe para compensar 63 ASD vs 40 TD durante o backpropagation
    n_td = np.sum(y_suj == 0)
    n_asd = np.sum(y_suj == 1)
    pesos_classe = {0: len(y_suj) / (2.0 * n_td), 1: len(y_suj) / (2.0 * n_asd)}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_spec, y_suj)):
        fixar_sementes(config.RANDOM_STATE + fold)

        topo_tr, topo_te = X_topo[train_idx], X_topo[test_idx]
        spec_tr, spec_te = X_spec[train_idx], X_spec[test_idx]
        y_tr, y_te = y_suj[train_idx], y_suj[test_idx]

        scaler = RobustScaler()
        spec_tr_sc = scaler.fit_transform(spec_tr)
        spec_te_sc = scaler.transform(spec_te)

        selector = SelectKBest(f_classif, k=k_feats)
        spec_tr_sel = selector.fit_transform(spec_tr_sc, y_tr)
        spec_te_sel = selector.transform(spec_te_sc)

        model = criar_rede_convolucional_hibrida(n_canais, k_feats)
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=18, restore_best_weights=True, verbose=0
        )

        model.fit(
            [topo_tr, spec_tr_sel], y_tr,
            validation_data=([topo_te, spec_te_sel], y_te),
            epochs=120,
            batch_size=16,
            class_weight=pesos_classe,
            callbacks=[early_stop],
            verbose=0
        )

        prob_tr = model.predict([topo_tr, spec_tr_sel], verbose=0).ravel()
        melhor_th = 0.50
        maior_j = -1.0
        for th in np.linspace(0.30, 0.70, 41):
            j_score = balanced_accuracy_score(y_tr, (prob_tr >= th).astype(int))
            if j_score > maior_j:
                maior_j = j_score
                melhor_th = th

        prob_te = model.predict([topo_te, spec_te_sel], verbose=0).ravel()
        preds_te = (prob_te >= melhor_th).astype(int)

        acertos_fold = np.sum(preds_te == y_te)
        print(f"Fold {fold+1}/5: Avaliou {len(y_te)} pacientes -> Acertou {acertos_fold} ({acertos_fold/len(y_te):.2%})")

        y_true_all.extend(y_te)
        y_pred_all.extend(preds_te)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc_bruta = np.mean(y_true_all == y_pred_all)
    acc_bal = balanced_accuracy_score(y_true_all, y_pred_all)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)

    print("\n=======================================================")
    print("RESULTADO FINAL DA FASE 2: DEEP LEARNING (BRAINNET-CNN)")
    print("=======================================================")
    print(f"Diagnósticos Corretos:        {tn+tp}/{len(y_suj)} ({acc_bruta:.2%})")
    print(f"Acurácia Clínica Balanceada:  {acc_bal:.2%}")
    print(f"Acerto em Neurotípicos (TD):  {tn}/{tn+fp} ({spec:.2%})")
    print(f"Acerto em Autismo      (ASD): {tp}/{tp+fn} ({sens:.2%})")
    print("=======================================================\n")

if __name__ == '__main__':
    executar_treinamento_dl()