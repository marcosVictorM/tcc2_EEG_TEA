import matplotlib.pyplot as plt
import numpy as np

def plotar_matriz_confusao(cm, titulo, arquivo_saida, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax)

    classes = ['Neurotipico (TD)', 'Autismo (ASD)']
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[f'{c} (Pred)' for c in classes],
        yticklabels=[f'{c} (Real)' for c in classes],
        title=titulo,
        ylabel='Diagnostico Clinico Real',
        xlabel='Previsao do Modelo'
    )
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=18, fontweight='bold'
            )

    fig.tight_layout()
    plt.savefig(arquivo_saida, dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    # 1. Fase 3.2: BrainNet-CNN Isolada em REST (73.79% Bruta | 74.92% Balanceada)
    cm_dl = np.array([[32, 8],
                      [19, 44]])
    plotar_matriz_confusao(
        cm_dl,
        'Matriz de Confusao: BrainNet-CNN Isolada (REST) - 74.92% Bal.',
        'matriz_confusao_fase3_2_youden.png',
        cmap='Blues'
    )

    # 2. Fase 4 (Campeao 1): Super-Fusao Multi-Paradigma (82.52% Acuracia Bruta | 92.06% Sens. ASD)
    cm_rec = np.array([[27, 13],
                       [5, 58]])
    plotar_matriz_confusao(
        cm_rec,
        'Super-Fusao Multi-Paradigma: 82.52% Acuracia (85/103) | 92.06% Sens. ASD',
        'matriz_confusao_fase4_super_fusao_82_5.png',
        cmap='Greens'
    )

    # 3. Fase 4 (Campeao 2): Super-Fusao Clinica Balanceada (80.60% Bal. | 85.00% Espec. TD)
    cm_bal = np.array([[34, 6],
                       [15, 48]])
    plotar_matriz_confusao(
        cm_bal,
        'Super-Fusao Clinica Balanceada: 80.60% Bal. (82/103) | 85.00% Espec. TD',
        'matriz_confusao_fase4_balanceada_80_6.png',
        cmap='Purples'
    )

    print("Graficos da Fase 3.2 e Fase 4 gerados com sucesso!")
