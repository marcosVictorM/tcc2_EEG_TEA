# 02_treinamento_ml.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import config

def validacao_cruzada_robusta():
    print("Iniciando Fase 1.3: Validação Cruzada K-Fold com Gradient Boosting...")
    
    if not os.path.exists(config.X_FEATURES_PATH):
        print("Erro: Arquivos .npy não encontrados. Rode 'python 01_extracao.py' primeiro.")
        return

    X = np.load(config.X_FEATURES_PATH)
    y = np.load(config.Y_LABELS_PATH)
    df_meta = pd.read_csv(config.METADATA_PATH)
    
    participant_ids = df_meta['participante_id'].values
    arquivos_epocas = df_meta['arquivo'].values

    # Identificar pacientes únicos e labels reais para estratificação
    pacientes_unicos = df_meta[['participante_id', 'arquivo']].drop_duplicates()
    labels_unicos = [y[participant_ids == pid][0] for pid in pacientes_unicos['participante_id']]

    print(f"Dataset carregado: {X.shape[0]} épocas × {X.shape[1]} features de {len(pacientes_unicos)} participantes.")
    print("\nIniciando 5-Fold Cross Validation (Todos os pacientes serão testados de forma rotativa)...\n")

    # Configuração do Pipeline (Os melhores parâmetros descobertos nos testes anteriores)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=config.RANDOM_STATE)), 
        ('clf', HistGradientBoostingClassifier(
            max_iter=200, 
            max_depth=7, 
            learning_rate=0.01, 
            l2_regularization=1.0,
            random_state=config.RANDOM_STATE
        ))
    ])

    # K-Fold com estratificação (garante proporção de TEA/Neuro em cada rodada)
    cv_externo = StratifiedGroupKFold(n_splits=5)
    
    acertos_globais = 0
    total_pacientes = 0
    
    # Execução rotativa
    for fold, (train_idx, test_idx) in enumerate(cv_externo.split(X, y, groups=participant_ids)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        grupos_teste = participant_ids[test_idx]
        arquivos_teste_fold = arquivos_epocas[test_idx]
        
        # Treina o pipeline nesta configuração específica
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        # Avaliação Clínica Majoritária para este Fold
        pids_teste = np.unique(grupos_teste)
        acertos_fold = 0
        
        for pid in pids_teste:
            mask_pid = grupos_teste == pid
            pred_pid = y_pred[mask_pid]
            true_pid = y_test[mask_pid][0]
            
            voto_majoritario = int(pred_pid.mean() >= 0.5)
            if voto_majoritario == true_pid:
                acertos_fold += 1
                acertos_globais += 1
                
        total_pacientes += len(pids_teste)
        taxa_fold = acertos_fold / len(pids_teste)
        print(f"Fold {fold+1}/5: Avaliou {len(pids_teste)} pacientes -> Acertou {acertos_fold} ({taxa_fold:.2%})")

    taxa_global = acertos_globais / total_pacientes
    print("\n=======================================================")
    print(f"RESULTADO FINAL DO MODELO DE MACHINE LEARNING CLÁSSICO")
    print("=======================================================")
    print(f"Total de Pacientes Avaliados: {total_pacientes} (56 pacientes únicos rotacionados)")
    print(f"Diagnósticos Clínicos Corretos: {acertos_globais}")
    print(f"Acurácia Global Robusta: {taxa_global:.2%}")
    print("=======================================================\n")

if __name__ == '__main__':
    validacao_cruzada_robusta()