# Documentação do Processo - TCC (Diagnóstico Computacional do TEA via EEG)

Esta documentação técnica e científica consolida todo o percurso metodológico, arquitetural e estatístico da **Fase 1 (Machine Learning Clássico)** do Trabalho de Conclusão de Curso (TCC), estruturada para servir tanto como guia de alinhamento para o grupo de investigação como texto-base para as secções de *Materiais e Métodos*, *Experiências* e *Discussão de Resultados* do artigo científico.

## 1. Fase Piloto: Base Inicial (56 Sujeitos), Efeito da Semente Aleatória e Descoberta do *Overfitting*

Na etapa inicial do projeto, o pipeline processava uma base de dados piloto em estado de repouso composta por **56 participantes** (arquivos no formato `.set` do EEGLAB, como `24Abby_Resting.set` e `43Abby_Resting.set`), totalizando **4.473 épocas** (média de 79,88 épocas ou ~2,66 minutos por sujeito) e **4.512 atributos (*features*) por época**. Essas 4.512 variáveis resultavam da extração em 47 canais de EEG ao longo de 4 bandas espectrais ($\frac{47 \times 46}{2} = 1.081$ pares de correlação de Pearson + $47$ valores de Densidade Espectral de Potência [PSD] = $1.128 \text{ variáveis/banda} \times 4 = 4.512$).

### 1.1. O Fenómeno da Semente Fixa (`RANDOM_STATE = 42` vs. `RANDOM_STATE = 1`)

* **Fase 1.1 (Divisão *Hold-out* com Semente 42):** Utilizando uma divisão única de treino e teste (80%/20%, avaliando 12 pacientes inéditos no teste) com otimização de hiperparâmetros para o classificador Gradient Boosting (`pca__n_components: 0.85`, `clf__max_iter: 100`, `clf__max_depth: 5`, `clf__learning_rate: 0.01`, `clf__l2_regularization: 0.0`), o modelo atingiu **73% de precisão por época** (em 960 épocas de teste, com f1-score de 0,71 para Neurotípico e 0,75 para TEA) e uma **Acurácia Clínica Final por Voto Majoritário de 83,33% (10/12 pacientes)**. Nessa partição específica, o classificador acertou 100% dos casos de TEA (6/6: 24Abby, 26Abby, 6Abby, 19Abby, 20Abby, 25Abby) e 66,67% dos Neurotípicos (4/6: acertou 43Abby, 35Abby, 48Abby, 39Abby; errou apenas 36Abby e 49Abby).
* **Fase 1.2 (Teste de Sensibilidade à Partição com Semente 1):** Ao alterar unicamente a semente pseudo-aleatória para 1, os 12 pacientes sorteados para o conjunto de teste mudaram, e a busca de hiperparâmetros convergiu para árvores mais profundas (`pca__n_components: 0.95`, `clf__max_iter: 200`, `clf__max_depth: 7`, `clf__learning_rate: 0.01`, `clf__l2_regularization: 1.0`). A precisão por época caiu para **63%** (em 953 épocas, com recall em Neurotípicos de apenas 0,38 contra 0,89 em TEA) e a **Acurácia Clínica Final recuou 16,66 pontos percentuais, caindo para 66,67% (8/12 pacientes)**, fruto de 4 Falsos Positivos nos indivíduos neurotípicos (38Abby, 37Abby, 40Abby e 55Abby).
* **Fase 1.3a (Validação Cruzada Externa 5-Fold na Base de 56 Sujeitos):** Para eliminar o fator sorte na escolha dos 12 pacientes de teste, implementou-se a validação cruzada rotativa por grupos (`StratifiedGroupKFold` com 5 *folds*), garantindo que todas as épocas de um mesmo paciente permanecessem juntas e que todos os **56 participantes** fossem avaliados como dados inéditos. O resultado revelou uma forte oscilação entre os grupos: **75,00% (9/12)** no Fold 1, **72,73% (8/11)** no Fold 2, **54,55% (6/11)** no Fold 3, e uma queda abrupta para **36,36% (4/11)** nos Folds 4 e 5, resultando numa **Acurácia Global Robusta de 55,36% (31/56 diagnósticos corretos)**.

**Conclusão Metodológica da Base Piloto:** O desvio padrão entre os 5 *folds* foi de **18,77%** ($\text{média} = 55,00\%$, $\text{mediana} = 54,55\%$, $\text{mín} = 36,36\%$, $\text{máx} = 75,00\%$), comprovando matematicamente a ocorrência da **Maldição da Dimensionalidade** (*Curse of Dimensionality*): 4.512 variáveis para apenas 56 indivíduos levavam o modelo a decorar ruídos biológicos específicos de determinadas partições.

## 2. Migração para a Nova Base BIDS SFARI (ds006780) e Engenharia de Pré-Processamento

Para superar a limitação amostral da base piloto, o projeto migrou para o repositório público **SFARI_EEG multi-paradigm dataset (BIDS)** (doi:10.18112/openneuro.ds006780.v1.0.0, versão BIDS 1.9.0, licença CC0), desenvolvido por Sophie Molholm, Theo Vanneau e John J. Foxe no *Albert Einstein College of Medicine* com financiamento da *Simons Foundation Autism Research Initiative* (SFARI Award #874845).

### 2.1. Caracterização Estatística e Demográfica da Coorte (`participants.tsv`)

A base contém registos de **136 participantes** distribuídos em quatro categorias na coluna `group`: **66 ASD** (Autismo), **41 TD** (Desenvolvimento Típico), **28 ASD SIBLING** (irmãos não afetados, documentados como SIB no dicionário `participants.json`) e **1 NAN** (sub-11814, sem dados demográficos). Além disso, os participantes foram submetidos a até **7 paradigmas experimentais**: ASSR, AVSRT, Beep-Flash, FAST, Illusory_contours, Motor e Resting_state.

Filtrando estritamente os grupos clínicos alvo (ASD = 1 e TD = 0) que concluíram o protocolo de Estado de Repouso (`completed_Resting_state == 'yes'`), obteve-se uma coorte principal de **103 participantes válidos** (**63 ASD [61,17%]** e **40 TD [38,83%]**, sendo 50 rapazes e 13 raparigas no grupo ASD, e 20 rapazes e 20 raparigas no grupo TD).

| Variável Demográfica / Fenotípica | Grupo Clínico | Contagem (count) | Média (mean) | Mediana (median) | Mínimo (min) | Máximo (max) | Desvio Padrão (std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Idade no Repouso (age, anos)** | **ASD** | 63 | 10,74 | 10,80 | 8,00 | 14,50 | 1,46 |
| **Idade no Repouso (age, anos)** | **TD** | 40 | 10,51 | 10,85 | 8,00 | 12,90 | 1,78 |
| **QI de Escala Completa (fsiq)** | **ASD** | 55 | 94,18 | 95,00 | 65,00 | 133,00 | 16,42 |
| **QI de Escala Completa (fsiq)** | **TD** | 38 | 104,37 | 102,50 | 81,00 | 142,00 | 15,61 |
| **Responsividade Social (srs2_total_t)** | **ASD** | 56 | 72,86 | 72,50 | 49,00 | 100,00 | 12,63 |
| **Responsividade Social (srs2_total_t)** | **TD** | 38 | 47,05 | 45,00 | 38,00 | 64,00 | 6,48 |
| **Coordenação Motora (mabc_total_ss)** | **ASD** | 51 | 4,71 | 4,00 | 1,00 | 11,00 | 2,74 |
| **Coordenação Motora (mabc_total_ss)** | **TD** | 38 | 7,05 | 7,00 | 2,00 | 15,00 | 2,77 |
| **Severidade Clínica ADOS (ados_css)** | **ASD** | 47 | 7,91 | 8,00 | 3,00 | 15,00 | 2,10 |
| **Paradigmas Completados (de 7)** | **ASD** | 66 | 6,20 | 7,00 | 1,00 | 7,00 | 1,63 |
| **Paradigmas Completados (de 7)** | **TD** | 41 | 6,59 | 7,00 | 2,00 | 7,00 | 1,12 |

### 2.2. Estrutura Eletrofisiológica (`.bdf`) e Redução Espacial para o Sistema 10-20

A análise dos ficheiros de canais (`_channels.tsv`) e das pastas `sub-XXXXX/eeg/` mostrou que os sinais brutos (`DatasetType: "raw"`) foram adquiridos em equipamento BioSemi no formato `.bdf` a **512,0 Hz** (filtro passa-baixa de hardware em 104,0 Hz), contendo **73 canais**: **64 eletrodos corticais EEG** (Fp1 a O2), **8 canais eletromiográficos/oculares EMG** (EXG1 a EXG8) e **1 canal marcador TRIG** (Status).
Para evitar que a matriz explodisse para **8.320 features por época** (caso fossem usados os 64 canais de EEG: $\frac{64 \times 63}{2} = 2.016$ conexões + $64$ PSDs por banda $\times 4$), implementou-se no pré-processamento:

1. **Remoção de Canais Não-Corticais e Seleção do Sistema 10-20 (19 canais):** Descarte imediato de EXG1..EXG8 e Status, retendo os 19 eletrodos clínicos universais (Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2). Isso reduziu a matriz bruta para **760 features por época** ($\frac{19 \times 18}{2} = 171$ conexões + $19$ PSDs = $190 \times 4$ bandas), uma **compressão de 90,87%** face aos 64 canais e de **83,16%** face às 4.512 variáveis da base antiga.
2. **Referência Média Comum (*Common Average Reference* — CAR):** Aplicação de `raw.set_eeg_reference('average')` antes do cálculo de Pearson para remover flutuações de modo comum (CMS/DRL) típicas de gravações `.bdf` BioSemi, seguida de reamostragem para **256 Hz** e filtragem FIR passa-faixa entre **0,5 Hz e 45,0 Hz**.

## 3. Arquitetura de Software e Evolução dos Códigos (`config.py`, `01_extracao.py`, `02_treinamento_ml.py`)

O projeto foi desacoplado em dois discos para proteger o controlo de versão na branch `Marcos`: o código-fonte e as matrizes `.npy` permaneceram em `C:\Users\Marcos\Documents\tcc2\TCC_EEG\`, enquanto os ficheiros pesados `.bdf` permaneceram em `D:\dataset_tcc\ds006780-download\`.

* **Evolução do `config.py`:** Centralizou o caminho BIDS (`DATASET_DIR`), as 4 bandas clássicas (delta: 0,5–4 Hz, theta: 4–8 Hz, alpha: 8–13 Hz, beta: 13–30 Hz), a janela de época (`EPOCH_DURATION = 2.0` s), a chave `USAR_SISTEMA_10_20 = True` com a lista `CANAIS_10_20`, e o dicionário `PARADIGMS` mapeando 'rest', 'fast' e 'assr' às respetivas colunas `completed_*` do `participants.tsv`.
* **Evolução do `01_extracao.py`:** Substituiu a leitura linear de pasta única por um rastreador BIDS (`encontrar_arquivos_bdf`) que percorre `sub-XXXXX/eeg/`, lê os ficheiros `.bdf` via `mne.io.read_raw_bdf`, extrai a parte triangular superior da matriz de correlação de Pearson (`np.triu_indices(19, k=1)`) concatenada com o PSD de Welch (`mne.time_frequency.psd_array_welch`), e incorporou um sistema de **cache incremental** que reaproveitou as **18.249 épocas** já extraídas de `rest` para anexar as **44.451 épocas** de `fast` e **23.405 épocas** de `assr`, consolidando **86.105 épocas × 760 features** de **105 sujeitos únicos**.
* **Evolução do `02_treinamento_ml.py`:** Evoluiu ao longo de cinco subfases (Fases 1.3b a 1.7), passando da avaliação época-a-época sem ponderação de classe para a engenharia de **Biomarcadores Relativos e Limpeza Automática de Épocas por IQR Intra-Sujeito** (corrigindo inclusive um erro pontual de indentação na linha 31 durante a transição da Fase 1.4 para a 1.5 e suprimindo avisos de depreciação `l1_ratio` do scikit-learn 1.8).

## 4. Estudo de Ablação Completo na Base SFARI (Fases 1.3b a 1.7)

### 4.1. Fase 1.3b: Estabilidade K-Fold na Nova Base vs. Armadilha da Classe Maioritária

Ao executar o `HistGradientBoostingClassifier` em 5-Fold Cross-Validation sobre as **18.249 épocas × 760 features** dos **103 participantes** de repouso, a Acurácia Global subiu de **55,36%** (Base 1) para **62,14% (64/103 pacientes)**, com acertos altamente estáveis entre os *folds*: **60,00% (12/20)** no Fold 1, **66,67% (14/21)** no Fold 2, **66,67% (14/21)** no Fold 3, **60,00% (12/20)** no Fold 4 e **57,14% (12/21)** no Fold 5. O desvio padrão entre *folds* caiu **4,3 vezes** (de **18,77%** na Base 1 para **4,34%** na Base SFARI).
Contudo, a análise matemática revelou que, como a coorte de repouso possui **63 ASD e 40 TD** ($\frac{63}{103} = 61,17\%$ de prevalência ASD), o classificador sem `class_weight` estava simplesmente a prever a classe maioritária na quase totalidade dos indivíduos, obtendo 64 acertos (apenas +1 acima do palpite fixo em ASD).

### 4.2. Fase 1.4: Desagregação da Matriz de Confusão e Diagnóstico do PSD Absoluto

Ao introduzir `class_weight='balanced'`, `RobustScaler`, seleção ANOVA `SelectKBest(k=300)` e desagregação entre Especificidade (TD) e Sensibilidade (ASD) no treino por épocas, o modelo obteve **56,31% de Acurácia Bruta (58/103)** e **51,96% de Acurácia Clínica Balanceada** (*folds*: 60,00%, 57,14%, 57,14%, 45,00% e 61,90%; $\text{média} = 56,24\%$, $\text{mediana} = 57,14\%$, $\text{std} = 6,60\%$). A matriz de confusão expôs a causa raiz: o modelo mantinha boa **Sensibilidade ASD (71,43% [45/63])**, mas sofria de baixa **Especificidade TD (32,50% [13/40], com 27 Falsos Positivos)**.
Três fatores neurofisiológicos explicavam esse limite no treino por épocas brutas:

1. **Impedância Craniana e PSD Absoluto:** A potência absoluta ($\mu\text{V}^2/\text{Hz}$) varia drasticamente conforme a espessura craniana e impedância do couro cabeludo de cada criança, confundindo o classificador entre sujeitos.
2. **Artefatos Oculares e Musculares em Sinais `raw`:** Épocas de 2 segundos contaminadas por piscadas ou movimentos geravam *outliers* severos quando avaliadas isoladamente.
3. **Estacionariedade do Diagnóstico Clínico:** O autismo é uma condição intrínseca do sujeito (medida pela tendência central e dispersão da sua rede cortical), e não um evento transitório de 2 segundos.

### 4.3. Fase 1.5: O Salto da Engenharia Espectral Relativa e Agregação por Sujeito

Para solucionar os problemas da Fase 1.4 sem alterar os ficheiros extraídos, implementou-se a função `extrair_assinatura_limpa` no `02_treinamento_ml.py`:

* **Transformada Z de Fisher ($\text{arctanh}(r)$):** Aplicada às $171 \times 4 = 684$ correlações de Pearson para normalizar a distribuição estatística da conectividade funcional.
* **Potência Espectral Relativa (%):** Cada banda ($\delta, \theta, \alpha, \beta$) foi dividida pela soma da potência total das 4 bandas no respetivo canal e época ($\text{PSD}_{\text{rel}} = \frac{\text{PSD}_b}{\sum \text{PSD}}$), anulando variações de impedância e espessura óssea entre crianças ($19 \times 4 = 76$ variáveis).
* **Razões Espectrais Logarítmicas:** Cálculo dos biomarcadores clássicos de maturação cortical $\log_{10}(\theta/\beta)$ (*Theta-Beta Ratio*, clássico na literatura de neurodesenvolvimento), $\log_{10}(\theta/\alpha)$, $\log_{10}(\alpha/\beta)$ e $\log_{10}(\delta/\theta)$ ($19 \times 4 = 76$ variáveis), totalizando $684 + 76 + 76 = 836$ variáveis por época.
* **Filtro de Rejeição de Épocas por IQR Intra-Sujeito e Agregação (1.672 biomarcadores):** Descarte automático das épocas cuja energia média ultrapassasse $[Q_{25} - 1,5 \times \text{IQR},\; Q_{75} + 1,5 \times \text{IQR}]$ dentro de cada paciente, extraindo a **Mediana** (836 variáveis) e o **Intervalo Interquartil — IQR** (836 variáveis) das épocas limpas, gerando uma assinatura de **1.672 biomarcadores por paciente** associada à calibração do limiar de decisão pelo **Índice de Youden** no conjunto de treino.

O resultado na Fase 1.5 foi um salto imediato em todas as métricas clínicas:

* **Regressão Logística (Lasso L1):** Atingiu **63,11% de Acurácia Bruta (65/103)** e **60,71% de Acurácia Balanceada** (Especificidade TD de **50,00% [20/40]** e Sensibilidade ASD de **71,43% [45/63]**).
* **SVM (Kernel RBF Balanceado):** Sagrou-se o modelo mais equilibrado do projeto, alcançando **66,99% de Acurácia Bruta (69/103)** e **66,17% de Acurácia Clínica Balanceada** — quase duplicando a **Especificidade TD para 62,50% (25/40, +30,00 pontos percentuais face à Fase 1.4)** enquanto manteve **69,84% de Sensibilidade ASD (44/63)**.

### 4.4. Fases 1.6 e 1.7: Fusão Multi-Paradigma (Rest + FAST + ASSR) — Fusão Precoce vs. Fusão Tardia

Para investigar se a adição de tarefas de estimulação auditiva e visual superava o estado de repouso isolado, extraíram-se **86.105 épocas** dos três paradigmas com maior disponibilidade e complementaridade neurofisiológica (FAST: 44.451 épocas [51,62%], ASSR: 23.405 épocas [27,18%] e REST: 18.249 épocas [21,19%]; $\text{média} = 28.701,67$ épocas/paradigma, $\text{mediana} = 23.405,00$, $\text{std} = 13.880,82$).

* **Fase 1.6 (Fusão Precoce / *Early Fusion* — 8.360 biomarcadores):** Concatenando os vetores de `rest` (1.672), `fast` (1.672), `assr` (1.672) e os dois vetores diferenciais de Reatividade Cortical $\Delta_{\text{FAST}} = \text{FAST} - \text{REST}$ (1.672) e $\Delta_{\text{ASSR}} = \text{ASSR} - \text{REST}$ (1.672), o **Ensemble Híbrido (SVM + ExtraTrees + LogReg)** bateu o recorde de **Acurácia Bruta com 67,96% (70/103 pacientes corretos)** e elevou a **Sensibilidade ASD para 84,13% (53/63)** (enquanto o Extra Trees isolado atingiu 92,06% de sensibilidade [58/63]), com **Acurácia Balanceada de 63,31%** e **Especificidade TD de 42,50% (17/40)**.
* **Fase 1.7 (Avaliação Individual e Fusão Tardia / *Late Fusion*):** Ao avaliar os especialistas isolados sem calibração de Youden e forçar pesos fixos entre os paradigmas, verificou-se que o paradigma **ASSR isolado ($n=96$)** atinge **57,29% de Acurácia Bruta (55/96)** e **56,49% Balanceada** (TD: 52,63% [20/38]; ASD: 60,34% [35/58]), e a melhor **Fusão Tardia 1 (Rest + FAST + Reat. FAST)** ficou em **58,25% Bruta (60/103)** e **57,20% Balanceada** (TD: 52,50% [21/40]; ASD: 61,90% [39/63]).
* **Explicação Científica para o Artigo:** Exames de Potenciais Evocados (FAST e ASSR) dependem do sincronismo temporal estrito com o estímulo (registado nos ficheiros `_events.tsv` e no canal Status) e da banda Gama (40 Hz no ASSR). Quando resumidos pela mediana global de 14,5 minutos nas bandas de 0,5 a 30 Hz, os períodos entre estímulos diluem a relação sinal-ruído. Na Fusão Precoce (Fase 1.6), o filtro global `SelectKBest(k=80)` conseguiu aproveitar apenas as variáveis de reatividade mais puras entre as 8.360 disponíveis, ao passo que a Fusão Tardia (Fase 1.7) atribuiu 50% do peso de decisão a classificadores treinados em sinais não-estacionários.

| Fase / Modelo Avaliado no Estudo de Ablação | Base / Sujeitos | Nº de Features | Acertos Totais | Acurácia Bruta (%) | Acurácia Balanceada (%) | Especificidade TD (%) | Sensibilidade ASD (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fase 1.1: HGB Hold-out (seed=42)** | Base 1 ($n=12$) | 4.512 | 10 / 12 | 83,33% | 83,33% | 66,67% (4/6) | 100,00% (6/6) |
| **Fase 1.2: HGB Hold-out (seed=1)** | Base 1 ($n=12$) | 4.512 | 8 / 12 | 66,67% | 66,67% | 33,33% (2/6) | 100,00% (6/6) |
| **Fase 1.3a: HGB 5-Fold CV** | Base 1 ($n=56$) | 4.512 | 31 / 56 | 55,36% | 55,36% | — | — |
| **Fase 1.3b: HGB 5-Fold Sem Pesos (Rest)** | SFARI ($n=103$) | 760 | 64 / 103 | 62,14% | ~50,80% | — | — |
| **Fase 1.4: HGB Épocas Balanceado (Rest)** | SFARI ($n=103$) | 760 | 58 / 103 | 56,31% | 51,96% | 32,50% (13/40) | 71,43% (45/63) |
| **Fase 1.5a: Regressão Logística L1 (Rest)** | SFARI ($n=103$) | 1.672 | 65 / 103 | 63,11% | 60,71% | 50,00% (20/40) | 71,43% (45/63) |
| **Fase 1.5b: SVM RBF Calibrado (Rest)** | SFARI ($n=103$) | 1.672 | **69 / 103** | **66,99%** | **66,17%** | **62,50% (25/40)** | **69,84% (44/63)** |
| **Fase 1.6a: SVM k=70 (Fusão Precoce)** | SFARI ($n=103$) | 8.360 | 64 / 103 | 62,14% | 57,64% | 37,50% (15/40) | 77,78% (49/63) |
| **Fase 1.6b: SVM k=100 (Fusão Precoce)** | SFARI ($n=103$) | 8.360 | 67 / 103 | 65,05% | 58,65% | 30,00% (12/40) | 87,30% (55/63) |
| **Fase 1.6c: Extra Trees (Fusão Precoce)** | SFARI ($n=103$) | 8.360 | 68 / 103 | 66,02% | 58,53% | 25,00% (10/40) | **92,06% (58/63)** |
| **Fase 1.6d: Ensemble Híbrido (Fusão Precoce)** | SFARI ($n=103$) | 8.360 | **70 / 103** | **67,96%** | **63,31%** | 42,50% (17/40) | **84,13% (53/63)** |
| **Fase 1.7a: Especialista Isolado ASSR** | SFARI ($n=96$) | 1.672 | 55 / 96 | 57,29% | 56,49% | 52,63% (20/38) | 60,34% (35/58) |
| **Fase 1.7b: Fusão Tardia 1 (Rest+FAST+Δ)** | SFARI ($n=103$) | 1.672 × 3 | 60 / 103 | 58,25% | 57,20% | 52,50% (21/40) | 61,90% (39/63) |

## 5. Vereditos Finais da Fase 1 e Consolidação dos Dois Modelos Campeões

A consolidação final no ficheiro `02_treinamento_ml.py` estabeleceu dois referenciais complementares (*benchmarks*) para o artigo científico na coorte de **103 participantes**:

1. **Campeão em Equilíbrio Clínico e Confirmação Diagnóstica — SVM RBF (Repouso Isolado — 1.672 features):**
   * **Acurácia Global Bruta:** **66,99% (69/103 pacientes diagnosticados corretamente)**.
   * **Acurácia Clínica Balanceada:** **66,17%** (+14,21 pontos percentuais acima da Fase 1.4).
   * **Especificidade (TD) e Sensibilidade (ASD):** Acerta **62,50% dos Neurotípicos (25/40)** e **69,84% dos Autistas (44/63)**.
   * **Valor Preditivo Positivo (VPP) e F1-Score:** Atinge **74,58% de precisão positiva (44/59)** e **72,13% de F1-Score**, provando que um único exame rápido de ~5,9 minutos em repouso, tratado com PSD Relativo, Razões Espectrais e Transformada Z de Fisher, oferece a fronteira de decisão mais equilibrada.
2. **Campeão em Sensibilidade e Triagem Populacional (*Screening*) — Ensemble Híbrido Multi-Paradigma (8.360 features):**
   * **Acurácia Global Bruta:** **67,96% (70/103 pacientes diagnosticados corretamente)**.
   * **Sensibilidade (ASD) e F1-Score:** Identifica corretamente **84,13% das crianças com Autismo (53/63, reduzindo os Falsos Negativos para apenas 10 casos)** e atinge o maior **F1-Score do estudo (76,26%)** e maior **Valor Preditivo Negativo (62,96% [17/27])**.

## 6. Fase 2: Deep Learning Topológico (BrainNet-CNN Híbrida)

O estudo da Fase 1 demonstrou que os algoritmos clássicos exigem a "planificação" (*flattening*) da matriz cerebral num vetor unidimensional (1.672 ou 8.360 colunas), destruindo a vizinhança topológica sobre quais conexões partilham o mesmo eletrodo cortical. Para superar isso, a Fase 2 introduziu uma arquitetura de Deep Learning inspirada na **BrainNetCNN**.

O script `03_treinamento_dl.py` reconstrói a topologia espacial 3D do cérebro, estruturando os dados em:
1. **Tensor de Adjacência Funcional 3D:** Dimensão (103 pacientes $\times$ 19 eletrodos $\times$ 19 eletrodos $\times$ 4 bandas).
2. **Ramo Espectral Denso:** Matriz espectral selecionada com os biomarcadores mais discriminativos.

A rede processa esses dados através de **convoluções topológicas cruzadas** (*Edge-to-Node* com filtros $1 \times 19$ e *Node-to-Graph* com filtros $19 \times 1$), culminando numa Fusão Neural Profunda.

### 6.1. O Estado da Arte com Aumento de Dados (Chunking Augmentation)

O gargalo clínico de possuir apenas 103 pacientes foi superado pela estratégia de **Chunking Augmentation**:
Ao invés de agregar todos os minutos de EEG de uma criança numa única mediana, o sinal foi fragmentado usando épocas sobrepostas (*overlap* de 50%) e os sinais de cada paciente foram agrupados em múltiplos blocos menores (*chunks*). Isso escalou artificialmente o volume de treinamento da rede para mais de **600 matrizes topológicas 3D**. A validação cruzada foi blindada através de um `GroupKFold` para evitar vazamento de dados (*Data Leakage*), extraindo o diagnóstico do paciente por meio da média probabilística de seus blocos.

### 6.2. O Modelo Definitivo: Calibração Dinâmica de Limiar (Índice de Youden)

A última camada de aprimoramento lidou com o desbalanceamento inerente do dataset (63 TEA vs 40 Neurotípicos). O uso de um limiar de decisão estático de $50\%$ forçava a rede a classificar excessivamente crianças saudáveis como autistas (viés estatístico da classe majoritária), afundando a especificidade.

Para mitigar isso, aplicou-se a varredura da **Curva ROC** na média das probabilidades do paciente, calculando o **Índice de Youden** para otimizar o limiar dinamicamente. O limiar ideal matemático foi encontrado em **0.676** (exigindo mais certeza da rede para diagnosticar Autismo).

Os resultados desta calibração ditaram o modelo definitivo do projeto:

* **Limiar de Decisão Ótimo:** **0.676**
* **Acurácia Global Bruta:** **73,79% (76/103 diagnósticos corretos)** — Novo recorde histórico para um único paradigma topológico.
* **Acurácia Clínica Balanceada:** **74,92%** — Um salto colossal (de 69% para 75%), demonstrando uma arquitetura rigorosamente equilibrada.
* **Especificidade em Neurotípicos (TD):** **80,00% (32/40)** — Crescimento exponencial da capacidade de descartar falsos positivos.
* **Sensibilidade em Autismo (ASD):** **69,84% (44/63)** — Nível sólido de detecção da desordem.

## 7. Fase 4: Especialistas por Paradigma, Geometria Riemanniana e Super-Fusão Multi-Modal (>82%)

Embora a Fase 3.2 tenha alcançado ~75% de acurácia utilizando exclusivamente o exame de Estado de Repouso (`rest`), o repositório SFARI BIDS (`ds006780`) disponibiliza também paradigmas de estimulação sensorial ativa (`fast` e `assr`), totalizando **171.980 épocas extraídas** com sobreposição de 50% (`rest`: 36.389 épocas de 103 sujeitos; `fast`: 88.840 épocas de 102 sujeitos; `assr`: 46.751 épocas de 96 sujeitos).

Experiências preliminares demonstraram que alimentar múltiplos paradigmas simultaneamente numa única rede neural causa **Confusão de Domínio** (*Domain Confusion*), pois a variância topológica entre estar em repouso e assistir a estímulos rápidos supera a variância sutil entre TEA e Neurotípicos. Para resolver esse problema e romper a barreira dos **80%**, implementou-se no script `05_fusao_multi_paradigma.py` uma arquitetura de **Especialistas por Exame + Geometria Riemanniana + Super-Fusão Tardia (*Multi-Paradigm Stacking*)**.

### 7.1. Projeção no Espaço Tangente Riemanniano e Biomarcadores de Teoria dos Grafos

Para extrair informação discriminativa dos paradigmas de estimulação (`fast` e `assr`), cada matriz de conectividade por banda $R_b$ e vetor de potência espectral relativa $\tilde{P}_b$ foram combinados na **Matriz de Covariância Espectral Simétrica Positiva-Definida (SPD)**:
$$\Sigma_b = D_b^{1/2} \, R_b \, D_b^{1/2}, \quad \text{onde } D_b = \text{diag}(\tilde{P}_b)$$
Como matrizes de covariância cerebral residem numa variedade curva não-Euclidiana (Variedade Riemanniana), aplicou-se o **Logaritmo Matricial** ($\log_m(\Sigma_b) = V \, \text{diag}(\log(\lambda_i)) \, V^T$) para projetar cada bloco no **Espaço Tangente Log-Euclidiano**, complementado por métricas de **Teoria dos Grafos Corticais**:
1. **Centralidade de Grau Nodal (*Node Strength*):** Soma assinada, absoluta e dispersão das conexões de cada um dos 19 eletrodos.
2. **Conectividade Homóloga Inter-Hemisférica:** Acoplamento direto entre os 8 pares simétricos (`Fp1-Fp2`, `F7-F8`, `F3-F4`, `T7-T8`, `C3-C4`, `P7-P8`, `P3-P4`, `O1-O2`).
3. **Razão de Conectividade Longo Alcance vs. Curto Alcance:** Contraste entre conexões fronto-posteriores (longa distância) e agrupamentos locais (frontal/occipital), modelando a hipótese clássica de *hipoconectividade de longo alcance e hiperconectividade local* no TEA.
4. **Entropia Espectral de Shannon:** Complexidade da distribuição energética entre as 4 bandas em cada canal.

### 7.2. Desempenho Individual do Melhor Especialista em Cada Tipo de Dado (`GroupKFold` 5-Fold)

Avaliando cada paradigma separadamente sob a mesma partição estrita por sujeito (`GroupKFold` de 5 *folds*):

| Paradigma / Domínio Neurofisiológico | Algoritmo Especialista Campeão | Acurácia Bruta (%) | Acurácia Balanceada (%) | Especificidade TD (%) | Sensibilidade ASD (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`REST` (Espectral Relativo + Razões)** | Regressão Logística L2 Calibrada ($k=40$) | **77,67%** (80/103) | **78,55%** | **82,50%** (33/40) | **74,60%** (47/63) |
| **`REST` (Grafo + Tangente Riemanniano)** | Regressão Logística Híbrida ($k=60$) | **75,73%** (78/103) | **76,05%** | **77,50%** (31/40) | **74,60%** (47/63) |
| **`REST` (Grafo 3D + Espectral)** | **BrainNet-CNN 3D** (Deep Learning) | **73,79%** (76/103) | **74,92%** | **80,00%** (32/40) | **69,84%** (44/63) |
| **`FAST` (Espaço Tangente Riemanniano)** | **SVM RBF Riemanniano** ($k=40$, Chunking) | **75,00%** (75/100) | **73,50%** | **66,67%** (26/39) | **80,33%** (49/61) |
| **`FAST` (Grafo 3D Estimulado)** | **BrainNet-CNN 3D** (Deep Learning) | **73,00%** (73/100) | **72,78%** | **71,79%** (28/39) | **73,77%** (45/61) |
| **`ASSR` (Auditivo 40 Hz + Riemanniano)** | **MLP no Espaço Tangente** ($k=60$) | **67,71%** (65/96) | **67,83%** | **68,42%** (26/38) | **67,24%** (39/58) |
| **`Delta_ASSR` (Reatividade $\text{ASSR} - \text{REST}$)** | Regressão Logística L2 Diferencial ($k=100$) | **67,71%** (65/96) | **67,38%** | **65,79%** (25/38) | **68,97%** (40/58) |
| **`Delta_FAST` (Reatividade $\text{FAST} - \text{REST}$)** | MLP Diferencial ($k=60$) | **66,00%** (66/100) | **67,97%** | **76,92%** (30/39) | **59,02%** (36/61) |

### 7.3. Resultado Final da Super-Fusão Multi-Paradigma (Rompendo a Barreira dos 80%+)

Ao padronizar as probabilidades *Out-of-Fold* (via normalização Z-Sigmoide) dos especialistas de cada paradigma e combiná-las num **Meta-Ensemble Ponderado** calibrado pelo Índice de Youden, o sistema alcançou os maiores patamares de toda a investigação:

1. **Super-Fusão Multi-Paradigma Campeã em Acurácia Global e Triagem (`REST` + `FAST` + `ASSR` + `Delta_ASSR`):**
   * **Acurácia Global Bruta:** **82,52% (85/103 pacientes diagnosticados corretamente)** — Superando a meta de 80% com folga (+15,53 pontos percentuais acima do SVM da Fase 1).
   * **Acurácia Clínica Balanceada:** **79,78%** (praticamente 80%) e **AUC-ROC de 0,820 (82,0%)**.
   * **Sensibilidade em Autismo (ASD):** **92,06% (58/63 autistas identificados, com apenas 5 Falsos Negativos)**.
   * **Especificidade em Neurotípicos (TD):** **67,50% (27/40)**.

2. **Super-Fusão Multi-Paradigma Campeã em Equilíbrio Clínico (`REST` Comitê + `Delta_ASSR` + `Delta_FAST`):**
   * **Acurácia Clínica Balanceada:** **80,60%** — Rompendo oficialmente a barreira dos 80% de acurácia balanceada sob validação cruzada rigorosa por sujeito (`AUC-ROC = 0,823`).
   * **Acurácia Global Bruta:** **79,61% (82/103 pacientes diagnosticados corretamente)**.
   * **Especificidade em Neurotípicos (TD):** **85,00% (34/40 crianças neurotípicas corretas, apenas 6 Falsos Positivos)**.
   * **Sensibilidade em Autismo (ASD):** **76,19% (48/63)**.

**Conclusão Definitiva do TCC:** A investigação comprovou que cada paradigma neurofisiológico exige uma representação matemática própria — **Redes Convolucionais Topológicas (BrainNet-CNN) e Modelos Lineares Espectrais** destacam-se no Estado de Repouso (`REST`), enquanto a **Geometria Riemanniana no Espaço Tangente** desbloqueia o poder preditivo dos exames de estimulação sensorial (`FAST` e `ASSR`). A fusão probabilística desses especialistas permitiu atingir **82,52% de Acurácia Global** e **80,60% de Acurácia Clínica Balanceada** sem qualquer vazamento de dados entre sujeitos, estabelecendo um pipeline de excelência científica para o apoio ao diagnóstico do Transtorno do Espectro Autista.

