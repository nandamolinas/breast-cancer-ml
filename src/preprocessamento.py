"""
Pré-processamento da base Breast Cancer (UCI ML Repository, id 14).

Fonte: https://archive.ics.uci.edu/dataset/14/breast+cancer
286 instâncias, 9 atributos categóricos + atributo alvo `Class`
(no-recurrence-events / recurrence-events). Há 9 valores ausentes
('?'): 8 em `node-caps` e 1 em `breast-quad`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Ordem das colunas no arquivo original breast-cancer.data
COLUNAS = [
    "Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps",
    "deg-malig", "breast", "breast-quad", "irradiat",
]
ALVO = "Class"
CLASSE_POSITIVA = "recurrence-events"
ATRIBUTOS = [c for c in COLUNAS if c != ALVO]

# Domínios válidos (conforme breast-cancer.names)
DOMINIOS = {
    "age": ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
    "menopause": ["lt40", "ge40", "premeno"],
    "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
                   "35-39", "40-44", "45-49", "50-54", "55-59"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20",
                  "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"],
    "node-caps": ["yes", "no"],
    "deg-malig": ["1", "2", "3"],
    "breast": ["left", "right"],
    "breast-quad": ["left_up", "left_low", "right_up", "right_low", "central"],
    "irradiat": ["yes", "no"],
}

# Atributos ordinais: faixas de valores viram o índice da faixa (preserva a ordem)
ORDINAIS = ["age", "tumor-size", "inv-nodes", "deg-malig"]
# Atributos binários: yes/no, left/right -> 1/0
BINARIOS = {"node-caps": "yes", "irradiat": "yes", "breast": "left"}
# Atributos nominais: one-hot
NOMINAIS = ["menopause", "breast-quad"]

CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "data" / "breast-cancer.data"


def carregar_dados(caminho=CAMINHO_PADRAO) -> pd.DataFrame:
    """Lê o arquivo no formato original da UCI ('?' = valor ausente)."""
    df = pd.read_csv(caminho, header=None, names=COLUNAS, dtype=str,
                     na_values="?", keep_default_na=False)
    return df.apply(lambda s: s.str.strip())


def validar_paciente(registro: dict) -> dict:
    """Confere se cada atributo informado pertence ao domínio da base.
    Atributos ausentes ou '?' são aceitos e serão imputados pelo modelo."""
    limpo = {}
    for att in ATRIBUTOS:
        v = registro.get(att)
        if v is None or str(v).strip() in ("", "?"):
            limpo[att] = np.nan
            continue
        v = str(v).strip().lower()
        if v not in DOMINIOS[att]:
            raise ValueError(f"Valor inválido para '{att}': {v!r}. "
                             f"Valores aceitos: {DOMINIOS[att]}")
        limpo[att] = v
    return limpo


def _codificar(df: pd.DataFrame) -> pd.DataFrame:
    """Converte ordinais e binários em números; nominais seguem como texto."""
    df = pd.DataFrame(df, columns=ATRIBUTOS).copy()
    for att in ORDINAIS:
        mapa = {v: i for i, v in enumerate(DOMINIOS[att])}
        df[att] = df[att].map(mapa).astype(float)
    for att, positivo in BINARIOS.items():
        df[att] = df[att].map(lambda v: np.nan if pd.isna(v) else float(v == positivo))
    for att in NOMINAIS:
        df[att] = df[att].astype(object)
    return df


def criar_preprocessador(escalonar: bool = True) -> Pipeline:
    """Pipeline de pré-processamento: codificação -> imputação (moda) ->
    padronização (numéricos) e one-hot (nominais). Fica dentro do pipeline
    do modelo, então é ajustado só com dados de treino em cada fold."""
    numericos = ORDINAIS + list(BINARIOS)
    passos_num = [("imputar", SimpleImputer(strategy="most_frequent"))]
    if escalonar:
        passos_num.append(("padronizar", StandardScaler()))
    colunas = ColumnTransformer([
        ("num", Pipeline(passos_num), numericos),
        ("nom", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), NOMINAIS),
    ])
    return Pipeline([
        ("codificar", FunctionTransformer(_codificar, validate=False)),
        ("colunas", colunas),
    ])
