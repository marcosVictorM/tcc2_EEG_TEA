# config.py
import os

# ==========================================
# CAMINHOS DE DIRETÓRIOS E ARQUIVOS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
LABELS_FILE = os.path.join(DATASET_DIR, 'labels_map.csv')

# Caminhos para salvar as matrizes extraídas (para agilizar treinos futuros)
X_FEATURES_PATH = os.path.join(BASE_DIR, 'X_features.npy')
Y_LABELS_PATH = os.path.join(BASE_DIR, 'y_labels.npy')
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.csv') # Guarda quem é o dono de cada época

# ==========================================
# PARÂMETROS DO EEG E EXTRAÇÃO
# ==========================================
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30)
}
EPOCH_DURATION = 2.0
SFREQ_RESAMPLE = 256
N_CANAIS = 47

# ==========================================
# PARÂMETROS DE MACHINE LEARNING
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.2