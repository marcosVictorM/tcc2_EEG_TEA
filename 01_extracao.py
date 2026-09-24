# 01_extracao.py
import os
import numpy as np
import pandas as pd
import mne
import config

def infer_label(setname):
    setname = str(setname)
    if setname.startswith('ASD'): return 1
    elif setname.startswith('P'): return 0
    return -1

def preprocess_raw(filepath):
    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
    raw.resample(config.SFREQ_RESAMPLE, verbose=False)
    raw.filter(0.5, 45.0, fir_design='firwin', verbose=False)
    raw.pick(raw.ch_names[:config.N_CANAIS])
    return raw

def extract_features(raw):
    """Extrai matrizes de Conectividade (Pearson) e Potência (PSD) por banda"""
    n_channels = len(raw.ch_names)
    band_features = []
    
    for band_name, (fmin, fmax) in config.BANDS.items():
        # 1. Filtro da Banda Específica
        raw_band = raw.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
        epochs_band = mne.make_fixed_length_epochs(
            raw_band, duration=config.EPOCH_DURATION, overlap=0.0, preload=True, verbose=False
        )
        data_band = epochs_band.get_data()
        
        # 2. Cálculo do PSD (Power Spectral Density) usando o método de Welch
        # O MNE calcula o espectro para cada época e cada canal
        psds, freqs = mne.time_frequency.psd_array_welch(
            data_band, sfreq=config.SFREQ_RESAMPLE, fmin=fmin, fmax=fmax, n_fft=256, verbose=False
        )
        # Tira a média da potência dentro da banda (resultado: 1 valor por canal por época)
        psd_mean = np.mean(psds, axis=2) 
        
        # 3. Iterar pelas épocas para extrair a Conectividade e concatenar com o PSD
        feats_this_band = []
        for ep_idx in range(data_band.shape[0]):
            epoch = data_band[ep_idx]
            
            # Conectividade (Correlação de Pearson)
            corr_matrix = np.corrcoef(epoch)
            upper_idx = np.triu_indices(n_channels, k=1)
            conectividade = corr_matrix[upper_idx]
            
            # Potência PSD para esta época específica
            potencia_psd = psd_mean[ep_idx]
            
            # Concatenar Conectividade + PSD no mesmo vetor de características
            epoca_features = np.concatenate([conectividade, potencia_psd])
            feats_this_band.append(epoca_features)
            
        band_features.append(np.array(feats_this_band))
        
    return np.concatenate(band_features, axis=1)

def executar_extracao():
    print("Iniciando Fase 1: Extração de Features Avançada (Conectividade + PSD)...")
    
    df_labels = pd.read_csv(config.LABELS_FILE)
    df_labels['label'] = df_labels['setname'].apply(infer_label)
    
    X_list, y_list = [], []
    metadata_list = []
    
    for idx, row in df_labels.iterrows():
        fname = row['arquivo']
        label = row['label']
        fpath = os.path.join(config.DATASET_DIR, fname)
        
        if not os.path.exists(fpath):
            print(f"[AVISO] Arquivo ausente: {fname}")
            continue

        try:
            print(f"Processando {fname} ({idx+1}/{len(df_labels)})...", end=" ")
            raw = preprocess_raw(fpath)
            # Chama a nossa nova função unificada
            feats = extract_features(raw) 
            
            X_list.append(feats)
            y_list.append(np.full(len(feats), label))
            
            for _ in range(len(feats)):
                metadata_list.append({'arquivo': fname, 'participante_id': idx})
                
            print(f"OK ({len(feats)} épocas)")
            
        except Exception as e:
            print(f"ERRO: {e}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    df_meta = pd.DataFrame(metadata_list)
    
    np.save(config.X_FEATURES_PATH, X)
    np.save(config.Y_LABELS_PATH, y)
    df_meta.to_csv(config.METADATA_PATH, index=False)
    
    print(f"\nExtração concluída com sucesso!")
    print(f"Total de épocas extraídas: {X.shape[0]}")
    print(f"Features por época: {X.shape[1]} (Pearson + PSD)")
    print(f"Arquivos salvos em: {config.BASE_DIR}")

if __name__ == '__main__':
    executar_extracao()