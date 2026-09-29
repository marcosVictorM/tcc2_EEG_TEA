# 01_extracao.py
import os
import numpy as np
import pandas as pd
import mne
import config

def encontrar_arquivos_bdf(participant_id, task_keyword):
    eeg_dir = os.path.join(config.DATASET_DIR, participant_id, 'eeg')
    if not os.path.exists(eeg_dir):
        return []
    
    arquivos_encontrados = []
    for fname in sorted(os.listdir(eeg_dir)):
        if fname.endswith('_eeg.bdf') and task_keyword.lower() in fname.lower():
            arquivos_encontrados.append(os.path.join(eeg_dir, fname))
    return arquivos_encontrados

def preprocess_raw(filepath):
    raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
    
    if config.USAR_SISTEMA_10_20:
        canais_presentes = [ch for ch in config.CANAIS_10_20 if ch in raw.ch_names]
        raw.pick(canais_presentes)
    else:
        canais_eeg = [ch for ch in raw.ch_names if not ch.startswith('EXG') and ch != 'Status']
        raw.pick(canais_eeg[:64])

    raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.resample(config.SFREQ_RESAMPLE, verbose=False)
    raw.filter(0.5, 45.0, fir_design='firwin', verbose=False)
    return raw

def extract_features(raw):
    n_channels = len(raw.ch_names)
    band_features = []
    
    for band_name, (fmin, fmax) in config.BANDS.items():
        raw_band = raw.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
        epochs_band = mne.make_fixed_length_epochs(
            raw_band, duration=config.EPOCH_DURATION, overlap=0.0, preload=True, verbose=False
        )
        data_band = epochs_band.get_data()
        
        psds, _ = mne.time_frequency.psd_array_welch(
            data_band, sfreq=config.SFREQ_RESAMPLE, fmin=fmin, fmax=fmax, n_fft=256, verbose=False
        )
        psd_mean = np.mean(psds, axis=2)
        
        feats_this_band = []
        upper_idx = np.triu_indices(n_channels, k=1)
        
        for ep_idx in range(data_band.shape[0]):
            epoch = data_band[ep_idx]
            corr_matrix = np.corrcoef(epoch)
            conectividade = np.nan_to_num(corr_matrix[upper_idx], nan=0.0)
            potencia_psd = np.nan_to_num(psd_mean[ep_idx], nan=0.0)
            
            epoca_features = np.concatenate([conectividade, potencia_psd])
            feats_this_band.append(epoca_features)
            
        band_features.append(np.array(feats_this_band))
        
    return np.concatenate(band_features, axis=1)

def executar_extracao():
    print("Iniciando Extração Multi-Paradigma BIDS (SFARI_EEG ds006780)...")
    
    df_part = pd.read_csv(config.PARTICIPANTS_FILE, sep='\t')
    df_asd_td = df_part[df_part['group'].isin(['ASD', 'TD'])].copy().reset_index(drop=True)
    
    X_list, y_list, metadata_list = [], [], []
    paradigmas_ja_extraidos = set()

    # Reaproveitamento inteligente: se Resting State já foi extraído, mantém na memória
    if (os.path.exists(config.X_FEATURES_PATH) and 
        os.path.exists(config.Y_LABELS_PATH) and 
        os.path.exists(config.METADATA_PATH)):
        try:
            X_old = np.load(config.X_FEATURES_PATH)
            y_old = np.load(config.Y_LABELS_PATH)
            df_meta_old = pd.read_csv(config.METADATA_PATH)
            
            if 'paradigma' not in df_meta_old.columns:
                df_meta_old['paradigma'] = 'rest'
                
            X_list.append(X_old)
            y_list.append(y_old)
            metadata_list.extend(df_meta_old.to_dict('records'))
            paradigmas_ja_extraidos = set(df_meta_old['paradigma'].unique())
            print(f"[CACHE] Reaproveitando {len(X_old)} épocas já extraídas dos paradigmas: {paradigmas_ja_extraidos}")
        except Exception as e:
            print(f"[AVISO] Não foi possível reaproveitar cache anterior ({e}). Extraindo do zero...")
            X_list, y_list, metadata_list = [], [], []
            paradigmas_ja_extraidos = set()

    for p_key, (task_kw, completed_col) in config.PARADIGMS.items():
        if p_key in paradigmas_ja_extraidos:
            print(f"\n---> Paradigma '{p_key.upper()}' já consta nos arquivos locais. Pulando para o próximo...")
            continue
            
        df_validos = df_asd_td[df_asd_td[completed_col] == 'yes'].copy().reset_index(drop=True)
        print(f"\n---> Extraindo Paradigma '{p_key.upper()}' ({len(df_validos)} sujeitos elegíveis)...")
        
        suj_proc = 0
        for idx, row in df_validos.iterrows():
            pid = row['participant_id']
            label = 1 if row['group'] == 'ASD' else 0
            
            bdf_files = encontrar_arquivos_bdf(pid, task_kw)
            if not bdf_files:
                print(f"[AVISO] Nenhum arquivo .bdf com '{task_kw}' para {pid}.")
                continue
                
            try:
                epocas_sujeito = []
                for fpath in bdf_files:
                    raw = preprocess_raw(fpath)
                    feats = extract_features(raw)
                    epocas_sujeito.append(feats)
                    
                feats_totais = np.vstack(epocas_sujeito)
                X_list.append(feats_totais)
                y_list.append(np.full(len(feats_totais), label))
                
                for _ in range(len(feats_totais)):
                    metadata_list.append({
                        'arquivo': os.path.basename(bdf_files[0]),
                        'participante_id': pid,
                        'grupo': row['group'],
                        'paradigma': p_key
                    })
                    
                suj_proc += 1
                print(f"[{p_key.upper()} {suj_proc}/{len(df_validos)}] {pid} ({row['group']}) -> OK ({len(feats_totais)} épocas)")
            except Exception as e:
                print(f"[ERRO] {pid} ({p_key}): {e}")

    if not X_list:
        print("Nenhum dado extraído.")
        return

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    df_meta = pd.DataFrame(metadata_list)
    
    np.save(config.X_FEATURES_PATH, X)
    np.save(config.Y_LABELS_PATH, y)
    df_meta.to_csv(config.METADATA_PATH, index=False)
    
    print("\n=======================================================")
    print("EXTRAÇÃO MULTI-PARADIGMA CONCLUÍDA COM SUCESSO!")
    print(f"Total de sujeitos únicos: {df_meta['participante_id'].nunique()}")
    print(f"Épocas por paradigma:\n{df_meta['paradigma'].value_counts().to_string()}")
    print(f"Dimensão final da matriz X: {X.shape[0]} épocas × {X.shape[1]} features")
    print("=======================================================")

if __name__ == '__main__':
    executar_extracao()