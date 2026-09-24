# Diagnóstico Computacional do TEA via Eletroencefalografia (EEG)
 
**Contexto:** Trabalho de Conclusão de Curso (TCC) - Engenharia da Computação, PUC Minas.

## Visão Geral
Este repositório contém a implementação metodológica para a classificação automatizada do Transtorno do Espectro Autista (TEA) em comparação com o desenvolvimento Neurotípico, utilizando sinais de Eletroencefalograma (EEG) em estado de repouso. 

O objetivo central da pesquisa é avaliar e comparar a eficácia de pipelines tradicionais de Machine Learning (Feature Engineering + Classificadores) contra abordagens avançadas de Deep Learning na identificação de biomarcadores neurológicos.

## Base de Dados
A base de dados é composta por gravações de EEG de **56 participantes** (28 diagnosticados com TEA e 28 Neurotípicos de controle).
*   **Formato:** Arquivos no padrão EEGLAB (`.set` para metadados/canais e `.fdt` para os sinais brutos).
*   **Canais:** Sinais padronizados para os 47 canais espacialmente correspondentes entre todos os sujeitos.
*   *Nota: Por questões de privacidade médica, os arquivos de dados brutos (`/dataset`) estão incluídos no `.gitignore` e não são versionados neste repositório.*

## Metodologia Atual (Machine Learning Clássico)
O script principal (`main.py`) executa o seguinte pipeline:
1.  **Pré-processamento:** Leitura via biblioteca `mne`, reamostragem para 256Hz e filtragem *band-pass* (0.5Hz - 45.0Hz) para remoção de ruídos e artefatos de rede elétrica.
2.  **Segmentação:** Divisão do sinal contínuo em épocas não sobrepostas de 2 segundos.
3.  **Extração de Características:** Cálculo da matriz de conectividade funcional (Correlação de Pearson) isolada pelas frequências clássicas (Delta, Theta, Alpha, Beta).
4.  **Redução de Dimensionalidade:** Aplicação de PCA (Principal Component Analysis) para retenção de 85%-95% da variância explicada, mitigando a maldição da dimensionalidade.
5.  **Classificação:** Treinamento de modelos base (Random Forest, Regressão Logística, SVM) balanceados para lidar com a assimetria das classes.
6.  **Validação Rigorosa:** Avaliação via `StratifiedGroupKFold` para garantir o isolamento total das épocas de um mesmo paciente entre os conjuntos de treino e teste, prevenindo *data leakage*. O diagnóstico final por sujeito é definido por voto majoritário.

## Estrutura do Repositório
```text
TCC-EEG-TEA/
│
├── dataset/                  # (Ignorado no Git) Arquivos .set, .fdt e labels_map.csv
├── main.py                   # Script principal de extração e classificação tradicional
├── Plano_TCC_EEG_Marcos.md   # Roteiro metodológico e cronograma de testes do TCC
├── .gitignore                # Regras de exclusão de artefatos pesados e ambientes
└── README.md                 # Documentação do projeto