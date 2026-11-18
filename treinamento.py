"""
Módulo de TREINAMENTO — Breast Cancer (UCI id 14)

Etapas:
  1. Carga e análise exploratória (distribuição das classes, ausentes, domínios)
  2. Separação estratificada treino/teste (holdout 80/20)
  3. Pré-processamento dentro do pipeline (codificação ordinal/binária,
     imputação pela moda, padronização e one-hot), evitando vazamento de dados
  4. Linha de base (ZeroR: sempre a classe majoritária)
  5. Comparação de vários classificadores com validação cruzada estratificada
     repetida (10 folds x 3 repetições) no conjunto de treino
  6. Ajuste de hiperparâmetros (GridSearchCV) de cada classificador, métrica: acurácia
  7. Escolha do modelo de MELHOR ACURÁCIA na validação cruzada e avaliação no
     conjunto de teste (nunca usado antes)
  8. Retreino do modelo escolhido com todos os dados e gravação em models/

Uso:
    python treinamento.py [--dados data/breast-cancer.data] [--seed 42]
"""
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessamento import (ALVO, ATRIBUTOS, CLASSE_POSITIVA, CAMINHO_PADRAO,
                                  carregar_dados, criar_preprocessador)

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent
DIR_MODELOS = RAIZ / "models"
DIR_RELATORIOS = RAIZ / "reports"


def candidatos(seed):
    """Classificadores e grades de hiperparâmetros (prefixo 'clf__')."""
    return {
        "ZeroR (baseline)": (DummyClassifier(strategy="most_frequent"), {}),
        "Naive Bayes": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1, 1.0]}),
        "KNN": (KNeighborsClassifier(), {
            "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]}),
        "Árvore de Decisão": (DecisionTreeClassifier(random_state=seed), {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [2, 3, 4, 5, None],
            "clf__min_samples_leaf": [1, 5, 10, 20]}),
        "Regressão Logística": (LogisticRegression(max_iter=2000), {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__class_weight": [None, "balanced"]}),
        "SVM": (SVC(probability=True, random_state=seed), {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", 0.01, 0.1]}),
        "Random Forest": (RandomForestClassifier(random_state=seed, n_jobs=-1), {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__max_features": ["sqrt", 0.5]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=seed), {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [1, 2, 3]}),
        "AdaBoost": (AdaBoostClassifier(random_state=seed), {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.05, 0.1, 0.5, 1.0]}),
        "Rede Neural (MLP)": (MLPClassifier(max_iter=2000, early_stopping=False, random_state=seed), {
            "clf__hidden_layer_sizes": [(5,), (10,), (20,), (10, 5)],
            "clf__alpha": [1e-4, 1e-2, 1.0, 10.0]}),
    }


def analise_exploratoria(df, log):
    log("=" * 70)
    log("1. ANÁLISE EXPLORATÓRIA")
    log("=" * 70)
    log(f"Instâncias: {len(df)} | Atributos: {len(ATRIBUTOS)} + alvo '{ALVO}'")
    dist = df[ALVO].value_counts()
    log("\nDistribuição da classe:")
    for k, v in dist.items():
        log(f"  {k:<22} {v:>4}  ({v / len(df):.1%})")
    log("\nValores ausentes ('?') por atributo:")
    aus = df.isna().sum()
    for k, v in aus[aus > 0].items():
        log(f"  {k:<12} {v}")
    log("\nTaxa de recorrência por atributo (proporção de 'recurrence-events'):")
    y = (df[ALVO] == CLASSE_POSITIVA).astype(int)
    for att in ATRIBUTOS:
        taxa = y.groupby(df[att]).agg(["mean", "count"])
        partes = ", ".join(f"{i}={r['mean']:.0%}(n={int(r['count'])})" for i, r in taxa.iterrows())
        log(f"  {att:<12} {partes}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    dist.plot.bar(ax=ax, color=["#4C78A8", "#E45756"])
    ax.set_title("Distribuição da classe")
    ax.set_ylabel("Pacientes")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(DIR_RELATORIOS / "distribuicao_classes.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default=str(CAMINHO_PADRAO))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teste", type=float, default=0.2, help="fração do holdout de teste")
    args = ap.parse_args()

    DIR_MODELOS.mkdir(exist_ok=True)
    DIR_RELATORIOS.mkdir(exist_ok=True)
    linhas = []

    def log(msg=""):
        print(msg)
        linhas.append(msg)

    t0 = time.time()
    df = carregar_dados(args.dados)
    analise_exploratoria(df, log)

    X = df[ATRIBUTOS]
    y = (df[ALVO] == CLASSE_POSITIVA).astype(int)  # 1 = recurrence-events

    # ---- 2. Holdout estratificado ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.teste, stratify=y, random_state=args.seed)
    log("\n" + "=" * 70)
    log("2. DIVISÃO TREINO/TESTE (estratificada)")
    log("=" * 70)
    log(f"Treino: {len(X_tr)}  |  Teste: {len(X_te)}  "
        f"(recorrência: {y_tr.mean():.1%} treino, {y_te.mean():.1%} teste)")

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=args.seed)
    metricas = {"acc": "accuracy", "f1": "f1", "recall": "recall", "auc": "roc_auc"}

    # ---- 3/4/5. Comparação com hiperparâmetros padrão ----
    log("\n" + "=" * 70)
    log("3. COMPARAÇÃO DOS CLASSIFICADORES (padrão) — CV 10 folds x 3 repetições")
    log("=" * 70)
    modelos = candidatos(args.seed)
    resultados_padrao, acc_por_fold = [], {}
    for nome, (clf, _) in modelos.items():
        pipe = Pipeline([("prep", criar_preprocessador()), ("clf", clf)])
        r = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=metricas, n_jobs=-1)
        acc_por_fold[nome] = r["test_acc"]
        resultados_padrao.append({
            "modelo": nome, "acuracia": r["test_acc"].mean(), "desvio": r["test_acc"].std(),
            "f1": r["test_f1"].mean(), "recall": r["test_recall"].mean(), "auc": r["test_auc"].mean()})
    tab = pd.DataFrame(resultados_padrao).sort_values("acuracia", ascending=False)
    log(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 6. Ajuste de hiperparâmetros ----
    log("\n" + "=" * 70)
    log("4. AJUSTE DE HIPERPARÂMETROS (GridSearchCV, métrica = acurácia)")
    log("=" * 70)
    ajustados = []
    for nome, (clf, grade) in modelos.items():
        pipe = Pipeline([("prep", criar_preprocessador()), ("clf", clf)])
        gs = GridSearchCV(pipe, grade or {}, cv=cv, scoring="accuracy", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        i = gs.best_index_
        ajustados.append({
            "modelo": nome, "acuracia_cv": gs.best_score_,
            "desvio": gs.cv_results_["std_test_score"][i],
            "params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "estimador": gs.best_estimator_})
        log(f"{nome:<22} acc CV = {gs.best_score_:.4f} ± {gs.cv_results_['std_test_score'][i]:.4f}  "
            f"{ajustados[-1]['params']}")

    ajustados.sort(key=lambda d: d["acuracia_cv"], reverse=True)
    melhor = ajustados[0]
    base = next(a for a in ajustados if a["modelo"].startswith("ZeroR"))

    # ---- 7. Avaliação no teste ----
    log("\n" + "=" * 70)
    log("5. MODELO ESCOLHIDO E AVALIAÇÃO NO CONJUNTO DE TESTE")
    log("=" * 70)
    log(f"Melhor acurácia na validação cruzada: {melhor['modelo']} "
        f"({melhor['acuracia_cv']:.4f}) | baseline ZeroR: {base['acuracia_cv']:.4f}")
    log(f"Hiperparâmetros: {melhor['params']}")

    log("\nAcurácia no teste (todos os modelos ajustados):")
    teste = []
    for a in ajustados:
        acc = accuracy_score(y_te, a["estimador"].predict(X_te))
        teste.append({"modelo": a["modelo"], "acuracia_cv": a["acuracia_cv"], "acuracia_teste": acc})
    log(pd.DataFrame(teste).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    est = melhor["estimador"]
    pred = est.predict(X_te)
    proba = est.predict_proba(X_te)[:, 1]
    acc_te = accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred)
    log(f"\n>>> {melhor['modelo']} — acurácia no teste: {acc_te:.4f}")
    log(f"F1 (recorrência): {f1_score(y_te, pred):.4f} | Recall (recorrência): "
        f"{recall_score(y_te, pred):.4f} | AUC: {roc_auc_score(y_te, proba):.4f}")
    log("\nMatriz de confusão (linhas = real, colunas = previsto) [sem recorrência, recorrência]:")
    log(str(cm))
    log("\n" + classification_report(y_te, pred, target_names=["no-recurrence-events",
                                                                "recurrence-events"], digits=4))

    # Gráfico de comparação (boxplot das acurácias por fold, modelos padrão)
    ordem = tab["modelo"].tolist()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([acc_por_fold[m] for m in ordem], vert=False, tick_labels=ordem)
    ax.invert_yaxis()
    ax.axvline(base["acuracia_cv"], ls="--", color="gray", label="ZeroR")
    ax.set_xlabel("Acurácia (CV 10x3, hiperparâmetros padrão)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DIR_RELATORIOS / "comparacao_modelos.png", dpi=120)
    plt.close(fig)

    # ---- 8. Retreino com todos os dados e gravação ----
    final = clone(est).fit(X, y)
    meta = {
        "modelo": melhor["modelo"],
        "hiperparametros": melhor["params"],
        "acuracia_cv_treino": round(float(melhor["acuracia_cv"]), 4),
        "acuracia_teste_holdout": round(float(acc_te), 4),
        "acuracia_baseline_zeror": round(float(base["acuracia_cv"]), 4),
        "classes": {"0": "no-recurrence-events", "1": "recurrence-events"},
        "atributos": ATRIBUTOS,
        "instancias_treino_final": int(len(X)),
        "seed": args.seed,
        "sklearn": sklearn.__version__,
        "treinado_em": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump({"pipeline": final, "meta": meta}, DIR_MODELOS / "modelo_final.joblib")
    (DIR_MODELOS / "modelo_final.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    pd.DataFrame([{k: v for k, v in a.items() if k != "estimador"} for a in ajustados]) \
        .to_csv(DIR_RELATORIOS / "resultados_gridsearch.csv", index=False)

    log("\n" + "=" * 70)
    log(f"Modelo final ({melhor['modelo']}) retreinado com {len(X)} instâncias e salvo em "
        f"models/modelo_final.joblib  [{time.time() - t0:.0f}s]")
    (DIR_RELATORIOS / "relatorio_treinamento.txt").write_text("\n".join(linhas), encoding="utf-8")


if __name__ == "__main__":
    main()
