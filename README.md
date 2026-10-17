# Diagnóstico Computacional do TEA via Eletroencefalografia (EEG)

**Contexto:** Trabalho de Conclusão de Curso (TCC) — Engenharia da Computação, PUC Minas.

---

## 1. Visão Geral

Este repositório contém a implementação metodológica e experimental para a triagem e auxílio ao diagnóstico automatizado do **Transtorno do Espectro Autista (TEA / ASD)** em comparação com indivíduos **Neurotípicos (TD)**, a partir de sinais de Eletroencefalografia (EEG) multi-paradigma.

A pesquisa investiga e compara de forma progressiva:
1. **Machine Learning Clássico com Engenharia de Biomarcadores Relativos** (Potência Espectral Relativa, Razões Logarítmicas de Maturação Cortical, Transformada Z de Fisher e Limpeza Automática de Artefatos via IQR Intra-Sujeito).
2. **Deep Learning Topológico 3D (`BrainNet-CNN`)** com aumento de dados temporal (*Chunking Augmentation* + *Overlap* de 50%), preservando a geometria espacial dos 19 eletrodos do Sistema Internacional 10-20.
3. **Geometria Riemanniana no Espaço Tangente ($\text{SPD}$) + Teoria dos Grafos Corticais + Super-Fusão Multi-Paradigma (*Stacking*)**, treinando algoritmos especialistas para cada modalidade de exame (`REST`, `FAST`, `ASSR` e Reatividades Corticais $\Delta$) sob validação cruzada rigorosa por sujeito (`GroupKFold` de 5 *folds*, livre de *data leakage*).

---

## 2. Base de Dados (SFARI_EEG Multi-Paradigm BIDS — `ds006780`)

Após uma fase piloto inicial com 56 indivíduos (arquivos `.set`), o projeto migrou para a base clínica pública **SFARI_EEG multi-paradigm dataset (BIDS)**, desenvolvida no *Albert Einstein College of Medicine* e disponibilizada no repositório OpenNeuro:
* **Referência / DOI:** `doi:10.18112/openneuro.ds006780.v1.0.0` (Licença CC0).
* **Coorte Clínica Avaliada:** **103 participantes elegíveis** (**63 TEA [ASD]** e **40 Neurotípicos [TD]**, faixa etária de 8 a 14,5 anos).
* **Paradigmas Extraídos e Processados (171.980 épocas de 2s com 50% de sobreposição):**
  * **`rest` (*Resting State*):** Estado de repouso espontâneo (103 sujeitos | 36.389 épocas).
  * **`fast` (*FAST Stimulation*):** Estimulação sensorial auditiva/visual rápida (102 sujeitos | 88.840 épocas).
  * **`assr` (*Auditory Steady-State Response*):** Sincronização auditiva rítmica a 40 Hz (96 sujeitos | 46.751 épocas).
* **Padronização Espacial:** Redução de 73 canais brutos BioSemi (`.bdf` a 512 Hz) para os **19 canais corticais do Sistema 10-20** (`Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2`), com Referência Média Comum (*Common Average Reference* — CAR), reamostragem a **256 Hz** e filtragem FIR passa-faixa (**0,5 Hz – 45,0 Hz**).

---

## 3. Síntese dos Resultados Alcançados (Validação 5-Fold por Sujeito)

Todos os resultados abaixo foram avaliados estritamente no nível do paciente (`Subject-level GroupKFold` de 5 *folds*), garantindo que épocas ou blocos (*chunks*) de um mesmo participante **jamais** se misturem entre treino e teste:

| Fase / Arquitetura Avaliada | Paradigmas Utilizados | Acertos Totais | Acurácia Bruta (%) | Acurácia Balanceada (%) | Especificidade TD (%) | Sensibilidade ASD (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fase 1.4:** HGB em Épocas Brutas (PSD Absoluto) | `REST` | 58 / 103 | 56,31% | 51,96% | 32,50% (13/40) | 71,43% (45/63) |
| **Fase 1.5:** SVM RBF + Biomarcadores Relativos | `REST` | 69 / 103 | 66,99% | 66,17% | 62,50% (25/40) | 69,84% (44/63) |
| **Fase 1.6:** Ensemble Híbrido (Fusão Precoce) | `REST + FAST + ASSR` | 70 / 103 | 67,96% | 63,31% | 42,50% (17/40) | 84,13% (53/63) |
| **Fase 2.5:** Deep Learning `BrainNet-CNN` 3D (Limiar 0.50) | `REST` (6 *chunks*/suj) | 74 / 103 | 71,84% | 69,23% | 57,50% (23/40) | 80,95% (51/63) |
| **Fase 3.2:** `BrainNet-CNN` 3D + Calibração de Youden | `REST` (6 *chunks*/suj) | 76 / 103 | 73,79% | 74,92% | 80,00% (32/40) | 69,84% (44/63) |
| **Fase 4 (Especialista `FAST`):** SVM Riemanniano SPD | `FAST` Isolado | 75 / 100 | 75,00% | 73,50% | 66,67% (26/39) | 80,33% (49/61) |
| **Fase 4 (Especialista `REST`):** LogReg L2 Espectral | `REST` Isolado | 80 / 103 | 77,67% | 78,55% | 82,50% (33/40) | 74,60% (47/63) |
| **Fase 4 (Campeão Equilíbrio):** Super-Fusão Balanceada | `REST + Δ_ASSR + Δ_FAST` | **82 / 103** | **79,61%** | **80,60%** | **85,00% (34/40)** | **76,19% (48/63)** |
| **Fase 4 (Campeão Global):** Super-Fusão Multi-Paradigma | `REST + FAST + ASSR + Δ_ASSR` | **85 / 103** | **82,52%** | **79,78%** | **67,50% (27/40)** | **92,06% (58/63)** |

---

## 4. Estrutura do Repositório

```text
TCC_EEG/
│
├── config.py                                      # Configurações globais (caminhos, canais 10-20, bandas e parâmetros)
├── 01_extracao.py                                 # Pré-processamento MNE (.bdf) e extração multi-paradigma com cache
├── 02_treinamento_ml.py                           # Fase 1: Machine Learning Clássico (SVM RBF e Ensemble Precoce)
├── 03_treinamento_dl.py                           # Fase 2 e 3: Deep Learning Topológico 3D (BrainNet-CNN + Youden)
├── 04_graficos.py                                 # Geração das Matrizes de Confusão em alta resolução (.png)
├── 05_fusao_multi_paradigma.py                    # Fase 4: Especialistas por Paradigma, Geometria Riemanniana e Super-Fusão (>82%)
├── Documentacao_Processo.md                       # Relatório científico detalhado de todas as fases e ablações do TCC
├── Plano_TCC.md                                   # Planejamento metodológico inicial do projeto
├── matriz_confusao_fase2_5.png                    # Matriz de Confusão da Fase 2.5 (BrainNet-CNN sem calibração)
├── matriz_confusao_fase3_2_youden.png             # Matriz de Confusão da Fase 3.2 (BrainNet-CNN + Índice de Youden)
├── matriz_confusao_fase4_balanceada_80_6.png      # Matriz de Confusão da Fase 4 (Super-Fusão Balanceada: 80,60%)
├── matriz_confusao_fase4_super_fusao_82_5.png     # Matriz de Confusão da Fase 4 (Super-Fusão Global: 82,52%)
├── .gitignore                                     # Exclusão de ambientes virtuais (venv) e matrizes pesadas (.npy/.npz)
└── README.md                                      # Guia do projeto e instruções de reprodução
```

---

## 5. Como Rodar os Testes em Outra Máquina (Passo a Passo)

Qualquer integrante do grupo ou avaliador pode reproduzir todos os experimentos seguindo os passos abaixo:

### Passo 1: Clonar o Repositório e Criar o Ambiente Virtual (`venv`)
Certifique-se de possuir o **Python 3.10 ou 3.11** instalado. No terminal (PowerShell ou CMD), execute:

```powershell
git clone <URL_DO_REPOSITORIO>
cd TCC_EEG

# Criar ambiente virtual
python -m venv venv

# Ativar o ambiente virtual (Windows)
.\venv\Scripts\activate
# (No Linux/macOS use: source venv/bin/activate)
```

### Passo 2: Instalar as Dependências
Com o ambiente virtual `(venv)` ativo, instale os pacotes científicos necessários:

```powershell
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn mne tensorflow matplotlib seaborn
```

### Passo 3: Preparar os Dados (Duas Opções)

Como os arquivos brutos de EEG (`.bdf`) e as matrizes extraídas (`X_features.npy` com ~997 MB) excedem o limite do GitHub, eles estão listados no `.gitignore`. Você pode escolher **uma** das duas formas abaixo para obter os dados:

* **Opção A — Rápida (Recomendada para o Grupo):**
  Copie os arquivos já extraídos (`X_features.npy`, `y_labels.npy`, `metadata.csv` e, opcionalmente, `cache_multi_paradigma.npz`) diretamente para a pasta raiz do projeto `TCC_EEG/`. Com isso, **não é necessário** baixar os arquivos `.bdf` brutos nem rodar o `01_extracao.py`.

* **Opção B — Extração Completa do Zero (A partir da Base Bruta OpenNeuro):**
  1. Baixe o dataset **ds006780** no OpenNeuro (`https://openneuro.org/datasets/ds006780`).
  2. Abra o arquivo `config.py` e altere a variável `DATASET_DIR` (linha 10) para o caminho onde a pasta `ds006780-download` foi salva no seu computador:
     ```python
     DATASET_DIR = r'D:\caminho\para\ds006780-download'
     ```
  3. Execute o extrator (possui salvamento e cache incremental por paradigma):
     ```powershell
     python 01_extracao.py
     ```

### Passo 4: Executar os Modelos e Gerar os Gráficos

Com as matrizes na raiz do projeto, você pode rodar qualquer uma das fases de treinamento e validação:

1. **Rodar a Fase 1 (Machine Learning Clássico — SVM RBF e Ensemble Multi-Paradigma):**
   ```powershell
   python 02_treinamento_ml.py
   ```
2. **Rodar a Fase 3.2 (Deep Learning Topológico 3D — `BrainNet-CNN` + Calibração de Youden):**
   ```powershell
   python 03_treinamento_dl.py
   ```
3. **Rodar a Fase 4 (Especialistas por Paradigma + Geometria Riemanniana + Super-Fusão >82%):**
   ```powershell
   python 05_fusao_multi_paradigma.py
   ```
   *(Nota: Na primeira execução sem o arquivo `cache_multi_paradigma.npz`, o script construirá automaticamente o cache local a partir do `X_features.npy`)*
4. **Regerar os Gráficos de Matriz de Confusão (`.png`):**
   ```powershell
   python 04_graficos.py
   ```