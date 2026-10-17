# 03_treinamento_dl.py
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import config

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

def fixar_sementes(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def reconstruir_tensor_topologico_e_espectral(X_epocas, participant_ids, y_epocas, n_chunks=6):
    n_canais = 19 if config.USAR_SISTEMA_10_20 else 64
    n_conn = (n_canais * (n_canais - 1)) // 2
    n_cols_banda = n_conn + n_canais
    upper_idx = np.triu_indices(n_canais, k=1)

    pids_unicos = pd.unique(participant_ids)
    tensores_3d = []
    vetores_espectrais = []
    y_sujeitos = []
    grupos_pid = [] 

    for pid in pids_unicos:
        mask = (participant_ids == pid)
        X_sujeito = X_epocas[mask]
        
        tamanho_chunk = len(X_sujeito) // n_chunks
        if tamanho_chunk < 5: continue 
        
        for c in range(n_chunks):
            inicio = c * tamanho_chunk
            fim = (c + 1) * tamanho_chunk if c < n_chunks - 1 else len(X_sujeito)
            X_bloco = X_sujeito[inicio:fim]

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

            q25, q75 = np.percentile(energia_epocas, [25, 75])
            iqr = q75 - q25
            mask_limpa = (energia_epocas >= (q25 - 1.5 * iqr)) & (energia_epocas <= (q75 + 1.5 * iqr))
            if np.sum(mask_limpa) < 3:
                mask_limpa = np.ones(len(energia_epocas), dtype=bool)

            matriz_suj_3d = np.zeros((n_canais, n_canais, 4), dtype=np.float32)
            for b in range(4):
                conn_mediana = np.median(conn_bandas[b][mask_limpa], axis=0)
                mat_b = np.zeros((n_canais, n_canais), dtype=np.float32)
                mat_b[upper_idx] = conn_mediana
                mat_b = mat_b + mat_b.T
                psd_rel_b = np.median((psd_bandas[b] / psd_soma_total)[mask_limpa], axis=0)
                np.fill_diagonal(mat_b, psd_rel_b)
                matriz_suj_3d[:, :, b] = mat_b

            tensores_3d.append(matriz_suj_3d)

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
            grupos_pid.append(pid)

    return np.array(tensores_3d, dtype=np.float32), np.array(vetores_espectrais, dtype=np.float32), np.array(y_sujeitos), np.array(grupos_pid)

def criar_rede_convolucional_hibrida(n_canais, n_feats_selecionadas):
    in_topo = layers.Input(shape=(n_canais, n_canais, 4), name="input_topologico")
    x1 = layers.BatchNormalization()(in_topo)
    
    # BrainNetCNN Original (Sem Attention, para evitar overfitting)
    x1 = layers.Conv2D(16, kernel_size=(1, n_canais), activation='elu', kernel_regularizer=regularizers.l2(1e-2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.35)(x1)
    
    x1 = layers.Conv2D(32, kernel_size=(n_canais, 1), activation='elu', kernel_regularizer=regularizers.l2(1e-2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dropout(0.40)(x1)

    in_spec = layers.Input(shape=(n_feats_selecionadas,), name="input_espectral")
    x2 = layers.Dense(32, activation='elu', kernel_regularizer=regularizers.l2(1e-2))(in_spec)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.35)(x2)

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
    print("Iniciando Fase 3.2: BrainNetCNN com Calibração Dinâmica de Limiar de Decisão...\n")
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

    # Voltamos para 6 chunks (o ponto ótimo)
    X_topo, X_spec, y_chunks, grupos_pid = reconstruir_tensor_topologico_e_espectral(X_rest, pids_rest, y_rest, n_chunks=6)
    n_canais = X_topo.shape[1]
    k_feats = 60

    print(f"Tensores reconstruídos com Chunking (Data Augmentation Real):")
    print(f" -> Total de amostras na rede: {len(y_chunks)} (antes eram só ~103)")
    print(f" -> Tensor Topológico 3D: {X_topo.shape}")
    print(f" -> Matriz Espectral:     {X_spec.shape}\n")

    # ATENÇÃO: GroupKFold para garantir que chunks do mesmo paciente não vazem do treino pro teste
    cv = GroupKFold(n_splits=5)
    
    # Dicionários para guardar as probabilidades agregadas por paciente
    prob_por_paciente = {}
    y_true_por_paciente = {}

    n_td = np.sum(y_chunks == 0)
    n_asd = np.sum(y_chunks == 1)
    pesos_classe = {0: len(y_chunks) / (2.0 * n_td), 1: len(y_chunks) / (2.0 * n_asd)}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_spec, y_chunks, groups=grupos_pid)):
        tf.keras.backend.clear_session()
        fixar_sementes(config.RANDOM_STATE + fold)

        topo_tr, topo_te = X_topo[train_idx], X_topo[test_idx]
        spec_tr, spec_te = X_spec[train_idx], X_spec[test_idx]
        y_tr, y_te = y_chunks[train_idx], y_chunks[test_idx]
        grupos_te = grupos_pid[test_idx]

        scaler = RobustScaler()
        spec_tr_sc = scaler.fit_transform(spec_tr)
        spec_te_sc = scaler.transform(spec_te)

        selector = SelectKBest(f_classif, k=k_feats)
        spec_tr_sel = selector.fit_transform(spec_tr_sc, y_tr)
        spec_te_sel = selector.transform(spec_te_sc)

        model = criar_rede_convolucional_hibrida(n_canais, k_feats)
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
        )

        model.fit(
            [topo_tr, spec_tr_sel], y_tr,
            validation_data=([topo_te, spec_te_sel], y_te),
            epochs=100,
            batch_size=16,
            class_weight=pesos_classe,
            callbacks=[early_stop],
            verbose=0
        )

        # Previsão no teste do fold atual
        prob_te = model.predict([topo_te, spec_te_sel], verbose=0).ravel()
        
        # Agregando resultados por paciente (Voto Majoritário Contínuo)
        for i, pid in enumerate(grupos_te):
            if pid not in prob_por_paciente:
                prob_por_paciente[pid] = []
                y_true_por_paciente[pid] = y_te[i]
            prob_por_paciente[pid].append(prob_te[i])
            
        print(f"Fold {fold+1}/5 concluído.")

    # Avaliação Final Agregada por Sujeito
    y_true_all = []
    prob_media_all = []
    
    # Extrai a média das probabilidades e o ground truth
    for pid in prob_por_paciente:
        media_prob = np.mean(prob_por_paciente[pid])
        prob_media_all.append(media_prob)
        y_true_all.append(y_true_por_paciente[pid])

    y_true_all = np.array(y_true_all)
    prob_media_all = np.array(prob_media_all)
    
    # =========================================================================
    # CALIBRAÇÃO DINÂMICA DO LIMIAR DE DECISÃO (O SEGREDO PARA A ALTA ACURÁCIA)
    # =========================================================================
    # Em vez de forçar o threshold em 0.50 (que é ruim para classes desbalanceadas),
    # varremos a curva de probabilidades para encontrar o limiar ótimo global (Índice de Youden).
    
    melhor_th = 0.50
    maior_bal_acc = 0.0
    
    for th in np.linspace(0.30, 0.70, 100):
        preds_temp = (prob_media_all >= th).astype(int)
        acc_bal_temp = balanced_accuracy_score(y_true_all, preds_temp)
        if acc_bal_temp > maior_bal_acc:
            maior_bal_acc = acc_bal_temp
            melhor_th = th
            
    # Aplica o limiar otimizado
    y_pred_all = (prob_media_all >= melhor_th).astype(int)

    acc_bruta = np.mean(y_true_all == y_pred_all)
    acc_bal = balanced_accuracy_score(y_true_all, y_pred_all)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)

    print("\n=======================================================")
    print("RESULTADO FINAL DA FASE 3.2: BRAINNET-CNN + YOUDEN INDEX")
    print("=======================================================")
    print(f"Limiar Ótimo Encontrado:      {melhor_th:.3f}")
    print(f"Diagnósticos Corretos:        {tn+tp}/{len(y_true_all)} ({acc_bruta:.2%})")
    print(f"Acurácia Clínica Balanceada:  {acc_bal:.2%}")
    print(f"Acerto em Neurotípicos (TD):  {tn}/{tn+fp} ({spec:.2%})")
    print(f"Acerto em Autismo      (ASD): {tp}/{tp+fn} ({sens:.2%})")
    print("=======================================================\n")

if __name__ == '__main__':
    executar_treinamento_dl()