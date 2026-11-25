# Classificação de Transtorno do Espectro Autista (TEA) com EEG

Pipeline de pesquisa para distinguir indivíduos com TEA (ASD) de neurotípicos (TD) a partir de EEG, combinando **aprendizado de máquina clássico**, **redes neurais (BrainNetCNN híbrida)**, **geometria riemanniana** e **teoria de grafos**, sobre três paradigmas experimentais (repouso, FAST e ASSR). Trabalho desenvolvido como Trabalho de Conclusão de Curso (TCC).

> ⚠️ **Aviso:** este é um projeto de pesquisa acadêmica. Os modelos **não** são uma ferramenta de diagnóstico e não devem ser usados para decisões clínicas.

---

## Sumário

- [Dados](#dados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Visão geral do pipeline](#visão-geral-do-pipeline)
- [Instalação](#instalação)
- [Como executar](#como-executar)
- [Configuração](#configuração)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Limitações conhecidas](#limitações-conhecidas)
- [Reprodutibilidade](#reprodutibilidade)
- [Licença e citação](#licença-e-citação)

---

## Dados

O projeto usa o dataset público **SFARI EEG (OpenNeuro `ds006780`)**, em formato BIDS (arquivos `.bdf`, BioSemi). São usados apenas os participantes dos grupos `ASD` e `TD`, nos paradigmas:

| Paradigma | Descrição | Coluna em `participants.tsv` |
|---|---|---|
| `rest` | EEG em repouso | `completed_Resting_state` |
| `fast` | Paradigma FAST | `completed_FAST` |
| `assr` | Resposta auditiva em regime estacionário (ASSR, 40 Hz) | `completed_ASSR` |

**Os dados não estão incluídos neste repositório.** Baixe o dataset no OpenNeuro e ajuste `DATASET_DIR` em `config.py`.

Amostra de referência (repouso): 103 sujeitos, sendo 40 TD e 63 ASD (classes desbalanceadas).

---

## Estrutura do repositório

| Arquivo | Função |
|---|---|
| `config.py` | Caminhos, bandas, parâmetros de épocas, ICA, SMOTE e sementes |
| `01_extracao.py` | Pré-processamento (incluindo **ICA**) e extração de features por época |
| `02_treinamento_ml.py` | ML clássico (SVM, ensemble) em nível de sujeito, com métricas e correlação com o target |
| `03_treinamento_dl.py` | BrainNetCNN híbrida (tensor topológico 3D + vetor espectral), `GroupKFold` por sujeito |
| `04_graficos.py` | Matrizes de confusão (valores fixos no código; atualizar após novas execuções) |
| `05_fusao_multi_paradigma.py` | Especialistas por paradigma (espectral, grafos, espaço tangente riemanniano) e fusão multi-paradigma |
| `06_diagnostico_ica.py` | Diagnóstico do efeito do ICA (AUC antes/depois, sujeitos perdidos, features frontais) |
| `metricas.py` | Precisão, recall, F1, acurácia e análise de correlação feature × diagnóstico |
| `balanceamento.py` | SMOTE próprio (NumPy), aplicado somente no treino de cada fold |

---

## Visão geral do pipeline

```
BDF bruto
   │
   ├─ seleção de canais EEG → reamostragem 256 Hz → filtro 0.5–45 Hz → referência média
   ├─ ICA (ajuste em 1–40 Hz) → remoção de componentes oculares (proxy Fp1/Fp2)
   ├─ seleção de 19 canais (10-20) → referência média
   │
   ▼
Épocas de 2 s (50% de sobreposição) × 4 bandas (delta, theta, alpha, beta)
   │   conectividade (correlação, 171 pares) + PSD (19 canais) por banda
   ▼
X_features.npy / y_labels.npy / metadata.csv
   │
   ├─ 02: assinatura por sujeito (mediana + IQR) → SVM / ensemble
   ├─ 03: chunks por sujeito → tensor 19×19×4 + espectral → BrainNetCNN
   └─ 05: especialistas por paradigma + espaço tangente riemanniano → fusão
```

---

## Instalação

Requer **Python 3.10+**.

```bash
git clone <URL-DO-REPOSITORIO>
cd <NOME-DO-REPOSITORIO>
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

---

## Como executar

1. Baixe o dataset `ds006780` e ajuste `DATASET_DIR` em `config.py`.
2. Execute os scripts **na ordem**, na mesma pasta:

```bash
python 01_extracao.py              # gera X_features.npy, y_labels.npy, metadata.csv
python 02_treinamento_ml.py        # ML clássico + métricas + correlação
python 03_treinamento_dl.py        # BrainNetCNN
python 05_fusao_multi_paradigma.py # especialistas + fusão (gera cache_multi_paradigma.npz)
python 06_diagnostico_ica.py       # opcional: diagnóstico do ICA
python 04_graficos.py              # matrizes de confusão (atualize os valores antes)
```

**Importante sobre caches:** o `01_extracao.py` reaproveita `X_features.npy`, `y_labels.npy` e `metadata.csv` se já existirem, e o `05` reaproveita `cache_multi_paradigma.npz`. **Ao mudar o pré-processamento (por exemplo ligar/desligar o ICA), apague esses arquivos** antes de rodar novamente.

Para comparar com e sem ICA, renomeie os arquivos antigos para `X_features_sem_ica.npy`, `y_labels_sem_ica.npy` e `metadata_sem_ica.csv` antes de reextrair; o `06_diagnostico_ica.py` usa esses nomes.

---

## Configuração

Principais parâmetros em `config.py`:

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `DATASET_DIR` | `D:\dataset_tcc\ds006780-download` | Pasta do dataset BIDS |
| `EPOCH_DURATION` / `EPOCH_OVERLAP` | 2.0 / 1.0 s | Janelas de extração |
| `SFREQ_RESAMPLE` | 256 | Frequência de reamostragem (Hz) |
| `USAR_SISTEMA_10_20` | `True` | Usa 19 canais do sistema 10-20 |
| `USAR_ICA` | `True` | Remove artefatos oculares com ICA |
| `ICA_N_COMPONENTS` | 0.99 | Variância explicada (ou nº fixo de componentes) |
| `ICA_METHOD` | `fastica` | `fastica`, `infomax` ou `picard` |
| `ICA_EOG_CANAIS` | `['Fp1','Fp2']` | Canais de referência para detectar piscadas |
| `ICA_EOG_THRESHOLD` | 3.0 | Limiar z-score do MNE (menor = mais agressivo) |
| `ICA_MAX_EXCLUIR` | 3 | Máximo de componentes removidos por arquivo |
| `USAR_SMOTE` | `True` | Balanceamento com dados sintéticos (só no treino) |
| `SMOTE_K` | 5 | Vizinhos do SMOTE |
| `RANDOM_STATE` | 42 | Semente global |

---

## Metodologia

**Pré-processamento.** Reamostragem para 256 Hz, filtro passa-banda 0.5–45 Hz e referência média. O **ICA** é ajustado numa cópia filtrada em 1–40 Hz usando todos os canais EEG; componentes correlacionados com Fp1/Fp2 (proxy de EOG) são removidos do sinal original, e só então os 19 canais do sistema 10-20 são selecionados.

**Features por época.** Para cada uma das 4 bandas (delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30 Hz): correlação entre canais (171 pares) e potência espectral (PSD de Welch) por canal.

**Assinatura por sujeito (`02`).** Conectividade com transformação de Fisher (arctanh), PSD relativo e quatro razões espectrais (theta/beta, theta/alpha, alpha/beta, delta/theta). Épocas ruidosas são descartadas por IQR da energia; cada sujeito é resumido por **mediana + IQR** (1.672 features no repouso; 8.360 no multi-paradigma, combinando `rest`, `fast`, `fast−rest`, `assr` e `assr−rest`).

**Modelos.**
- `02`: SVM RBF (repouso) e ensemble soft voting SVM + ExtraTrees + Regressão Logística (multi-paradigma), com `RobustScaler` e `SelectKBest` dentro do pipeline.
- `03`: BrainNetCNN híbrida (convoluções 1×N e N×1 sobre o tensor 19×19×4 de conectividade/PSD + ramo denso espectral). Cada sujeito é dividido em 6 *chunks*; as probabilidades são agregadas por sujeito.
- `05`: especialistas por paradigma (espectral, grafos, espaço tangente riemanniano de matrizes SPD; LogReg, SVM, MLP) combinados por média ponderada.

**Validação.** `StratifiedKFold` por sujeito no `02` e `GroupKFold` por sujeito no `03` e `05`, evitando que dados do mesmo participante apareçam em treino e teste.

**Balanceamento.** Além do `class_weight='balanced'`, é aplicado SMOTE (interpolação entre vizinhos da classe minoritária) **apenas no conjunto de treino de cada fold**. Teste e métricas usam somente dados reais.

**Métricas.** Acurácia, acurácia balanceada, precisão, recall (sensibilidade), especificidade, F1 e AUC. Também é calculada a correlação de Pearson de cada feature com o diagnóstico, listando as features com |r| ≥ 0,7 (análise exploratória; não usada para selecionar features).

---

## Resultados

Resultados **preliminares**, de duas execuções do pipeline. Não são comparáveis diretamente (veja a nota abaixo da tabela).

| Execução | Sujeitos (REST) | AUC fusão | Acurácia fusão | Acurácia balanceada fusão |
|---|---|---|---|---|
| Sem ICA | 103 (40 TD / 63 ASD) | 0,820 – 0,823 | 79,6% – 82,5% | 79,8% – 80,6% |
| Com ICA + SMOTE | 93 (38 TD / 55 ASD) | 0,695 | 67,7% – 68,8% | 68,4% – 69,1% |

Especialistas isolados (acurácia bruta): REST 77,7% → 66,7%; FAST 75,0% → 62,5%; ASSR 67,7% → 52,3% (sem ICA → com ICA).

> **Nota:** entre as duas execuções, o número de sujeitos caiu de 103 para 93 (o pré-processamento com ICA falhou para alguns arquivos) e, além do ICA, foram introduzidos o SMOTE e novas métricas. A queda de desempenho, portanto, **não pode ser atribuída apenas ao ICA**. O `06_diagnostico_ica.py` compara as AUCs nos mesmos sujeitos para isolar o efeito do ICA. Uma hipótese em investigação é que artefatos oculares carregavam informação correlacionada ao grupo, inflando os resultados sem ICA.

*(Atualize esta seção com os resultados finais após recuperar os sujeitos perdidos e repetir a comparação nos mesmos sujeitos.)*

---

## Limitações conhecidas

- **Amostra pequena** (≈100 sujeitos): intervalos de confiança largos (≈ ±9 pontos na acurácia) e alta variabilidade entre partições. Para referência, prever sempre ASD já acerta ≈ 59–61%.
- **Otimismo nas estimativas:** nos scripts `03` e `05`, o limiar de decisão (índice de Youden) e os pesos da fusão são escolhidos sobre as mesmas previsões usadas na avaliação; no `03`, o *early stopping* usa o fold de teste como validação; e os "campeões" são selecionados entre vários candidatos. Para estimativas não enviesadas, use validação aninhada.
- **Pesos e textos fixos no `05` e valores fixos no `04`:** pesos da fusão, descrições ("RECORDE ...") e números das matrizes de confusão estão escritos no código e precisam ser atualizados após cada nova execução.
- **ICA com proxy de EOG:** sem canais oculares dedicados, o uso de Fp1/Fp2 pode remover parte da atividade cerebral frontal. Ajuste `ICA_EOG_CANAIS`, `ICA_EOG_THRESHOLD` e `ICA_MAX_EXCLUIR` conforme o dataset.
- **Confundidores:** idade, sexo, QI e qualidade do registro podem se correlacionar com o diagnóstico e não são controlados.
- **Sem validação externa:** nenhum teste em dataset independente.

---

## Reprodutibilidade

- Semente global `RANDOM_STATE = 42` (NumPy, TensorFlow, scikit-learn, ICA e SMOTE).
- A reprodutibilidade exata do ICA e do TensorFlow pode variar entre versões de bibliotecas e hardware.
- Registre as versões usadas (`pip freeze > requirements-lock.txt`).

---

## Licença e citação

- **Licença:** *(defina, por exemplo MIT)*.
- **Dataset:** respeite os termos de uso e cite o dataset `ds006780` (OpenNeuro / SFARI) conforme as instruções da página do dataset.
- **Autoria:** *(nome, instituição, orientador e ano)*.

```bibtex
@misc{tcc_eeg_tea,
  author = {<Autor>},
  title  = {Classificação de TEA com EEG: ML clássico, BrainNetCNN, geometria riemanniana e teoria de grafos},
  year   = {<ano>},
  note   = {Trabalho de Conclusão de Curso, <instituição>}
}
```
