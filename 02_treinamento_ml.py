# 02_treinamento_ml.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import config

def treinar_e_avaliar():
    print("Iniciando Fase 2: Carregando dados extraídos...")
    
    # Valida se a extração já foi feita
    if not os.path.exists(config.X_FEATURES_PATH):
        print("Erro: Arquivos .npy não encontrados. Rode 'python 01_extracao.py' primeiro.")
        return

    # Carregamento ultrarrápido
    X = np.load(config.X_FEATURES_PATH)
    y = np.load(config.Y_LABELS_PATH)
    df_meta = pd.read_csv(config.METADATA_PATH)
    
    participant_ids = df_meta['participante_id'].values
    arquivos_epocas = df_meta['arquivo'].values

    # Identificar quem são os pacientes únicos e seus respectivos labels (para o split stratificado)
    pacientes_unicos = df_meta[['participante_id', 'arquivo']].drop_duplicates()
    labels_unicos = []
    for pid in pacientes_unicos['participante_id']:
        labels_unicos.append(y[participant_ids == pid][0])

    print(f"Dataset carregado: {X.shape[0]} épocas × {X.shape[1]} features de {len(pacientes_unicos)} participantes.")

    # ==========================================
    # SPLIT TREINO/TESTE (Isolamento de Pacientes)
    # ==========================================
    part_train, part_test = train_test_split(
        pacientes_unicos['participante_id'].values, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=labels_unicos
    )

    mask_train = np.isin(participant_ids, part_train)
    mask_test  = np.isin(participant_ids, part_test)

    X_train, y_train, groups_train = X[mask_train], y[mask_train], participant_ids[mask_train]
    X_test, y_test, groups_test = X[mask_test], y[mask_test], participant_ids[mask_test]
    arquivos_teste = arquivos_epocas[mask_test]

    # ==========================================
    # PIPELINE E TUNING (BUSCA DE HIPERPARÂMETROS)
    # ==========================================
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=config.RANDOM_STATE)), 
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=config.RANDOM_STATE))
    ])

    param_dist = {
        'pca__n_components': [0.85, 0.90, 0.95],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    }

    cv_interno = StratifiedGroupKFold(n_splits=3)
    
    print("\nIniciando busca de hiperparâmetros (Tuning)...")
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=5, 
        cv=cv_interno, scoring='accuracy', random_state=config.RANDOM_STATE, n_jobs=-1
    )
    
    # O group_train garante que não há vazamento no CV interno
    search.fit(X_train, y_train, groups=groups_train)
    melhor_modelo = search.best_estimator_

    print(f"Melhores parâmetros encontrados: {search.best_params_}")

    # ==========================================
    # AVALIAÇÃO FINAL (VOTO MAJORITÁRIO)
    # ==========================================
    print("\n=== Relatório Final no Conjunto de Teste ===")
    y_pred = melhor_modelo.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Neurotípico', 'TEA']))

    print("\n=== Avaliação Clínica por Paciente ===")
    acertos = []
    
    for pid in part_test:
        mask_pid = groups_test == pid
        pred_pid = y_pred[mask_pid]
        true_pid = y_test[mask_pid][0]
        nome_arquivo = arquivos_teste[mask_pid][0]
        
        voto_majoritario = int(pred_pid.mean() >= 0.5)
        acerto = int(voto_majoritario == true_pid)
        acertos.append(acerto)
        
        status = "✅ ACERTOU" if acerto else "❌ ERROU"
        print(f"{status} | {nome_arquivo} | Diagnóstico Clínico: {true_pid} | Modelo: {voto_majoritario}")

    precisao_clinica = np.mean(acertos)
    print(f"\nAcurácia Real Final (Pacientes diagnosticados corretamente): {precisao_clinica:.2%} ({sum(acertos)}/{len(part_test)})")

if __name__ == '__main__':
    treinar_e_avaliar()