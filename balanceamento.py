# balanceamento.py
"""
Balanceamento de classes com dados sintéticos (SMOTE implementado em NumPy, sem depender do imbalanced-learn).

REGRA DE OURO: o SMOTE só pode atuar no TREINO de cada fold. Os dados de teste nunca recebem
amostras sintéticas, e as métricas são sempre calculadas só em dados reais.
Por isso o SMOTE entra:
  - como 'wrapper' do classificador final dos Pipelines (02 e 05): ele só age no .fit();
  - dentro do loop de folds do treino da rede neural (03), depois do scaler e do selector.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import config


def smote(X, y, k_neighbors=5, random_state=42):
    """Gera amostras sintéticas da classe minoritária até igualar a majoritária."""
    X = np.asarray(X)
    y = np.asarray(y)
    classes, contagens = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y, 0

    c_min = classes[np.argmin(contagens)]
    n_min, n_maj = contagens.min(), contagens.max()
    n_novos = int(n_maj - n_min)
    if n_novos <= 0 or n_min < 2:
        return X, y, 0

    rng = np.random.RandomState(random_state)
    X_min = X[y == c_min].astype(np.float64)
    k = min(k_neighbors, n_min - 1)

    # k vizinhos mais próximos (euclidiana) dentro da própria classe minoritária
    d2 = ((X_min[:, None, :] - X_min[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    vizinhos = np.argsort(d2, axis=1)[:, :k]

    base = rng.randint(0, n_min, size=n_novos)
    viz = vizinhos[base, rng.randint(0, k, size=n_novos)]
    lam = rng.rand(n_novos, 1)
    sinteticos = X_min[base] + lam * (X_min[viz] - X_min[base])

    X_out = np.vstack([X, sinteticos.astype(X.dtype)])
    y_out = np.concatenate([y, np.full(n_novos, c_min, dtype=y.dtype)])
    return X_out, y_out, n_novos


class ComSMOTE(BaseEstimator, ClassifierMixin):
    """Envolve qualquer classificador: aplica SMOTE no fit; predict/predict_proba usam dados reais."""

    def __init__(self, estimator, k_neighbors=5, random_state=42):
        self.estimator = estimator
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        Xb, yb, n_novos = smote(X, y, self.k_neighbors, self.random_state)
        self.n_sinteticos_ = n_novos
        self.estimator_ = clone(self.estimator).fit(Xb, yb)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def envolver_smote(clf, seed=None):
    """Retorna o classificador envolvido em SMOTE se config.USAR_SMOTE estiver ativo."""
    if not getattr(config, 'USAR_SMOTE', False) or isinstance(clf, ComSMOTE):
        return clf
    return ComSMOTE(clf, k_neighbors=config.SMOTE_K,
                    random_state=config.RANDOM_STATE if seed is None else seed)


def aplicar_smote_no_pipeline(pipe):
    """Troca o último passo do Pipeline por sua versão com SMOTE (só atua no fit)."""
    nome, est = pipe.steps[-1]
    pipe.steps[-1] = (nome, envolver_smote(est))
    return pipe


def balancear_treino_dl(topo_tr, spec_tr, y_tr, seed=42):
    """
    SMOTE para a rede híbrida (BrainNetCNN): interpola o tensor topológico (N,19,19,4)
    e o vetor espectral juntos, para os dois ramos continuarem consistentes entre si.
    """
    n = len(y_tr)
    flat = np.hstack([topo_tr.reshape(n, -1), spec_tr])
    flat_b, y_b, n_novos = smote(flat, y_tr, config.SMOTE_K, seed)
    n_topo = int(np.prod(topo_tr.shape[1:]))
    topo_b = flat_b[:, :n_topo].reshape((-1,) + topo_tr.shape[1:]).astype(topo_tr.dtype)
    spec_b = flat_b[:, n_topo:].astype(spec_tr.dtype)
    return topo_b, spec_b, y_b, n_novos
