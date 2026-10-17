Plano de Ação Metodológica - TCC

Fase 1: Otimização do Machine Learning Clássico

    O que vamos fazer: Melhorar o main.py atual.

    Como: Em vez de olhar apenas para a conectividade (Correlação de Pearson), vamos programar o MNE para extrair também a força das ondas cerebrais (PSD - Power Spectral Density). Depois, vamos trocar o Random Forest por algoritmos mais robustos como o Gradient Boosting ou Regressão com penalidade Lasso.

    Objetivo: Bater a marca dos 66% e ver qual é o limite máximo da abordagem "tradicional".

Fase 2: Transição para o Estado da Arte (Deep Learning)

    O que vamos fazer: Criar um script novo (main_deeplearning.py).

    Como: Vamos descartar a extração manual de matrizes e instalar o TensorFlow/Keras. Vamos construir uma EEGNet (uma Rede Neural Convolucional famosa na neurocomputação). A rede receberá o sinal bruto e aprenderá sozinha o que é autismo e o que é neurotípico.

    Objetivo: Provar se uma Inteligência Artificial avançada supera os métodos tradicionais num dataset de 56 pacientes.

Fase 3: Documentação e Versionamento

    Organizar ambos os scripts e as métricas para escrever os capítulos de "Metodologia" e "Resultados" da monografia.