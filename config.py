# config.py
import os

# ==========================================
# CAMINHOS DE DIRETÓRIOS E ARQUIVOS (BIDS)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho onde o dataset ds006780 foi baixado no Disco D:
DATASET_DIR = r'D:\dataset_tcc\ds006780-download'
PARTICIPANTS_FILE = os.path.join(DATASET_DIR, 'participants.tsv')

# Dicionário de Paradigmas para Fusão Multi-Paradigma:
# Chave = identificador curto | Valor = (palavra-chave no arquivo .bdf, coluna no participants.tsv)
PARADIGMS = {
    'rest': ('rest', 'completed_Resting_state'),
    'fast': ('fast', 'completed_FAST'),
    'assr': ('assr', 'completed_ASSR')
}

# Mantidos para retrocompatibilidade
TARGET_TASK_KEYWORD = 'rest'
COMPLETED_COL = 'completed_Resting_state'

# Caminhos para salvar as matrizes extraídas
X_FEATURES_PATH = os.path.join(BASE_DIR, 'X_features.npy')
Y_LABELS_PATH = os.path.join(BASE_DIR, 'y_labels.npy')
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.csv')

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

# Seleção de canais para evitar a Maldição da Dimensionalidade
USAR_SISTEMA_10_20 = True
CANAIS_10_20 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]

# ==========================================
# PARÂMETROS DE MACHINE LEARNING
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.2