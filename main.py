import os
import numpy as np
import pandas as pd
import mne
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ==========================================
# 1. CONFIGURAÇÕES GLOBAIS
# ==========================================
DATASET_DIR = './dataset'
LABELS_FILE = os.path.join(DATASET_DIR, 'labels_map.csv')

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30)
}
EPOCH_DURATION = 2.0
SFREQ_RESAMPLE = 256
RANDOM_STATE = 42

# Ajuste para a quantidade de canais que todos os pacientes de fato compartilham
N_CANAIS = 47 

# ==========================================
# 2. FUNÇÕES DE PRÉ-PROCESSAMENTO
# ==========================================
def infer_label(setname):
    """Mapeia ASD para 1 e P (Neurotípico) para 0"""
    setname = str(setname)
    if setname.startswith('ASD'): return 1
    elif setname.startswith('P'): return 0
    return -1

def preprocess_raw(filepath):
    """Lê, reamostra e filtra o sinal EEG"""
    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
    raw.resample(SFREQ_RESAMPLE, verbose=False)
    raw.filter(0.5, 45.0, fir_design='firwin', verbose=False)
    raw.pick(raw.ch_names[:N_CANAIS])
    return raw

def extract_connectivity_features(raw):
    """Extrai matrizes de correlação de Pearson por banda de frequência"""
    n_channels = len(raw.ch_names)
    band_features = []
    
    for band_name, (fmin, fmax) in BANDS.items():
        raw_band = raw.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
        epochs_band = mne.make_fixed_length_epochs(
            raw_band, duration=EPOCH_DURATION, overlap=0.0, preload=True, verbose=False
        )
        data_band = epochs_band.get_data()
        
        feats_this_band = []
        for ep_idx in range(data_band.shape[0]):
            epoch = data_band[ep_idx]
            corr_matrix = np.corrcoef(epoch)
            upper_idx = np.triu_indices(n_channels, k=1)
            feats_this_band.append(corr_matrix[upper_idx])
            
        band_features.append(np.array(feats_this_band))
        
    return np.concatenate(band_features, axis=1)

# ==========================================
# 3. EXTRAÇÃO DE CARACTERÍSTICAS
# ==========================================
def processar_dataset():
    df_labels = pd.read_csv(LABELS_FILE)
    df_labels['label'] = df_labels['setname'].apply(infer_label)
    
    X_list, y_list = [], []
    epocas_por_participante = []
    arquivos_processados = []

    print(f"Processando {len(df_labels)} participantes...")

    for _, row in df_labels.iterrows():
        fname = row['arquivo']
        label = row['label']
        fpath = os.path.join(DATASET_DIR, fname)
        
        if not os.path.exists(fpath):
            print(f"Arquivo não encontrado: {fname}")
            continue

        try:
            raw = preprocess_raw(fpath)
            feats = extract_connectivity_features(raw)
            
            X_list.append(feats)
            y_list.append(np.full(len(feats), label))
            epocas_por_participante.append(len(feats))
            arquivos_processados.append(fname)
            print(f"[OK] {fname}: {len(feats)} épocas extraídas.")
        except Exception as e:
            print(f"[ERRO] {fname}: {e}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y, epocas_por_participante, arquivos_processados, df_labels

# ==========================================
# 4. TREINAMENTO E VALIDAÇÃO
# ==========================================
if __name__ == '__main__':
    X, y, epocas_por_participante, arquivos_processados, df_labels = processar_dataset()

    print(f"\nDataset Final: {X.shape[0]} épocas × {X.shape[1]} features")

    # Criando IDs de participantes para agrupar as épocas corretamente no treino/teste
    participant_ids = []
    for i, n_ep in enumerate(epocas_por_participante):
        participant_ids.extend([i] * n_ep)
    participant_ids = np.array(participant_ids)

    # Recuperando o rótulo de cada participante
    arquivo_para_label = dict(zip(df_labels['arquivo'], df_labels['label']))
    participant_labels = np.array([arquivo_para_label[fname] for fname in arquivos_processados])
    participantes = np.arange(len(arquivos_processados))

    # Split separando participantes inéditos para o teste
    part_train, part_test = train_test_split(
        participantes, test_size=0.2, random_state=RANDOM_STATE, stratify=participant_labels
    )

    mask_train = np.isin(participant_ids, part_train)
    mask_test  = np.isin(participant_ids, part_test)

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test   = X[mask_test], y[mask_test]
    groups_train     = participant_ids[mask_train]
    groups_test      = participant_ids[mask_test]

    # Pipeline: Normalização -> Redução de Dimensionalidade -> Modelo com Peso Balanceado
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=RANDOM_STATE)), 
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE))
    ])

    # Grade de parâmetros para encontrar a melhor arquitetura
    param_dist = {
        'pca__n_components': [0.85, 0.90, 0.95],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    }

    cv_interno = StratifiedGroupKFold(n_splits=3)
    
    print("\nIniciando busca de hiperparâmetros (Random Forest)...")
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=5, 
        cv=cv_interno, scoring='accuracy', random_state=RANDOM_STATE, n_jobs=-1
    )
    
    search.fit(X_train, y_train, groups=groups_train)
    melhor_modelo = search.best_estimator_

    print(f"Melhores parâmetros encontrados: {search.best_params_}")

    # ==========================================
    # 5. AVALIAÇÃO FINAL
    # ==========================================
    y_pred = melhor_modelo.predict(X_test)
    print("\n--- Relatório Final no Conjunto de Teste ---")
    print(classification_report(y_test, y_pred, target_names=['Neurotípico', 'TEA']))

    print("\nAvaliação de Voto Majoritário por Participante:")
    acertos = []
    for g in part_test:
        mask_g = groups_test == g
        pred_g = y_pred[mask_g]
        true_g = y_test[mask_g][0]
        voto_majoritario = int(pred_g.mean() >= 0.5)
        acerto = int(voto_majoritario == true_g)
        acertos.append(acerto)
        
        resultado_texto = "ACERTOU" if acerto else "ERROU"
        print(f"[{resultado_texto}] {arquivos_processados[g]} | Real: {true_g} | Modelo: {voto_majoritario}")

    print(f"\nAcurácia real por paciente: {np.mean(acertos):.2%} ({sum(acertos)}/{len(part_test)})")