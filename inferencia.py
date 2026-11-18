"""
Módulo de INFERÊNCIA — Breast Cancer (UCI id 14)

Recebe os dados de uma nova paciente e devolve o indicativo do atributo
`Class`: no-recurrence-events ou recurrence-events (com a probabilidade).

Formas de uso:
  # 1) Argumentos na linha de comando
  python inferencia.py --age 40-49 --menopause premeno --tumor-size 15-19 \\
      --inv-nodes 0-2 --node-caps yes --deg-malig 3 --breast right \\
      --breast-quad left_up --irradiat no

  # 2) Interativo (pergunta cada atributo)
  python inferencia.py --interativo

  # 3) Arquivo JSON (uma paciente ou lista) ou CSV com cabeçalho
  python inferencia.py --arquivo exemplos/pacientes.json

  # 4) Como biblioteca
  from inferencia import prever
  prever({"age": "40-49", "menopause": "premeno", ...})

Atributos não informados (ou '?') são imputados pela moda da base de treino.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessamento import ATRIBUTOS, DOMINIOS, validar_paciente

CAMINHO_MODELO = Path(__file__).resolve().parent / "models" / "modelo_final.joblib"
_cache = {}


def carregar_modelo(caminho=CAMINHO_MODELO):
    caminho = Path(caminho)
    if caminho not in _cache:
        if not caminho.exists():
            sys.exit(f"Modelo não encontrado em {caminho}. Rode antes: python treinamento.py")
        _cache[caminho] = joblib.load(caminho)
    return _cache[caminho]


def prever(paciente, caminho_modelo=CAMINHO_MODELO):
    """Recebe um dict (ou lista de dicts) com os 9 atributos e devolve
    lista de dicts com 'Class' prevista e probabilidade de recorrência."""
    pacientes = [paciente] if isinstance(paciente, dict) else list(paciente)
    art = carregar_modelo(caminho_modelo)
    X = pd.DataFrame([validar_paciente(p) for p in pacientes], columns=ATRIBUTOS)
    pipe, classes = art["pipeline"], art["meta"]["classes"]
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return [{"Class": classes[str(int(c))],
             "prob_recurrence_events": round(float(p), 4),
             "entrada": {k: (None if pd.isna(v) else v) for k, v in x.items()}}
            for c, p, x in zip(pred, proba, X.to_dict("records"))]


def _ler_interativo():
    print("Informe os atributos da paciente (Enter = desconhecido):")
    paciente = {}
    for att in ATRIBUTOS:
        while True:
            v = input(f"  {att} {DOMINIOS[att]}: ").strip()
            if v == "" or v.lower() in DOMINIOS[att]:
                paciente[att] = v or None
                break
            print("    valor inválido, tente novamente.")
    return paciente


def _ler_arquivo(caminho):
    caminho = Path(caminho)
    if caminho.suffix.lower() == ".json":
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return dados if isinstance(dados, list) else [dados]
    return pd.read_csv(caminho, dtype=str).to_dict("records")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for att in ATRIBUTOS:
        ap.add_argument(f"--{att}", dest=att.replace("-", "_"), help=f"valores: {', '.join(DOMINIOS[att])}")
    ap.add_argument("--arquivo", help="JSON ou CSV com uma ou mais pacientes")
    ap.add_argument("--interativo", action="store_true")
    ap.add_argument("--modelo", default=str(CAMINHO_MODELO))
    ap.add_argument("--json", action="store_true", help="saída em JSON")
    args = ap.parse_args()

    if args.arquivo:
        pacientes = _ler_arquivo(args.arquivo)
    elif args.interativo:
        pacientes = [_ler_interativo()]
    else:
        p = {att: getattr(args, att.replace("-", "_")) for att in ATRIBUTOS}
        if all(v is None for v in p.values()):
            ap.error("informe os atributos, --arquivo ou --interativo")
        pacientes = [p]

    try:
        resultados = prever(pacientes, args.modelo)
    except ValueError as e:
        sys.exit(f"Erro: {e}")

    if args.json:
        print(json.dumps(resultados, indent=2, ensure_ascii=False))
        return
    meta = carregar_modelo(args.modelo)["meta"]
    print(f"Modelo: {meta['modelo']} (acurácia CV {meta['acuracia_cv_treino']:.1%}, "
          f"teste {meta['acuracia_teste_holdout']:.1%})")
    for i, r in enumerate(resultados, 1):
        print(f"\nPaciente {i}: {r['entrada']}")
        print(f"  Indicativo (Class): {r['Class']}")
        print(f"  Probabilidade de recorrência: {r['prob_recurrence_events']:.1%}")
    print("\nAviso: ferramenta acadêmica; não substitui avaliação médica.")


if __name__ == "__main__":
    main()
