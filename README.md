# Breast Cancer (UCI) — Treinamento e Inferência

Classificação do atributo **`Class`** (`no-recurrence-events` × `recurrence-events`) da base
[Breast Cancer — UCI ML Repository (id 14)](https://archive.ics.uci.edu/dataset/14/breast+cancer):
286 pacientes, 9 atributos categóricos, 9 valores ausentes.

## Estrutura

```
├── data/breast-cancer.data      # base no formato original da UCI ('?' = ausente)
├── src/preprocessamento.py      # leitura, domínios, validação e pipeline de pré-processamento
├── treinamento.py               # MÓDULO DE TREINAMENTO
├── inferencia.py                # MÓDULO DE INFERÊNCIA (nova paciente -> Class)
├── exemplos/pacientes.json      # pacientes de exemplo para a inferência
├── models/                      # modelo final (.joblib) + metadados (.json)
└── reports/                     # relatório, resultados do GridSearch e gráficos
```

## Instalação

```bash
pip install -r requirements.txt
```

## 1. Módulo de treinamento

```bash
python treinamento.py            # ~5 min; opções: --seed 42 --teste 0.2 --dados <arquivo>
```

Procedimentos executados:

1. **Análise exploratória** — distribuição da classe (70,3% sem recorrência × 29,7% com recorrência),
   valores ausentes (`node-caps`: 8, `breast-quad`: 1) e taxa de recorrência por valor de cada atributo.
2. **Holdout estratificado** 80/20 (228 treino / 58 teste). O teste só é usado no final.
3. **Pré-processamento dentro do pipeline** (ajustado apenas no treino de cada fold, sem vazamento):
   - faixas ordinais (`age`, `tumor-size`, `inv-nodes`) e `deg-malig` → índice ordinal;
   - binários (`node-caps`, `irradiat`, `breast`) → 0/1;
   - nominais (`menopause`, `breast-quad`) → one-hot;
   - ausentes → imputação pela moda; numéricos → padronização (z-score).
4. **Linha de base ZeroR** (sempre a classe majoritária) — qualquer modelo útil precisa superá-la.
5. **Comparação de 9 classificadores** com validação cruzada estratificada **10 folds × 3 repetições**:
   Naive Bayes, KNN, Árvore de Decisão, Regressão Logística, SVM, Random Forest, Gradient Boosting,
   AdaBoost e Rede Neural MLP (acurácia, F1, recall e AUC).
6. **Ajuste de hiperparâmetros** de todos os modelos com `GridSearchCV` otimizando a **acurácia**.
7. **Seleção do modelo de melhor acurácia** na validação cruzada e avaliação no teste
   (acurácia, matriz de confusão, precisão/recall/F1, AUC).
8. **Retreino** do modelo escolhido com as 286 instâncias e gravação em `models/modelo_final.joblib`.

### Resultados (seed 42)

| Modelo (após GridSearch) | Acurácia CV (treino) | Acurácia teste |
|---|---|---|
| **KNN** (k=21, manhattan) — **escolhido** | **0,7634** | 0,6724 |
| Random Forest | 0,7563 | 0,7241 |
| Rede Neural (MLP) | 0,7562 | 0,6897 |
| Árvore de Decisão | 0,7534 | 0,7414 |
| Gradient Boosting | 0,7446 | 0,7414 |
| SVM | 0,7403 | 0,6724 |
| AdaBoost | 0,7402 | 0,7586 |
| Regressão Logística | 0,7370 | 0,7069 |
| Naive Bayes | 0,7284 | 0,7069 |
| ZeroR (baseline) | 0,7020 | 0,7069 |

O relatório completo está em `reports/relatorio_treinamento.txt`.

**Observação.** Esta base é conhecidamente difícil: na literatura as acurácias ficam entre ~70% e ~78%,
pouco acima da baseline de 70,3%. Com só 58 pacientes no teste, cada erro vale ~1,7 ponto percentual, e
os desvios na validação cruzada (±5 a 7 p.p.) são maiores que as diferenças entre os modelos. Por isso a
escolha usa a acurácia média da validação cruzada repetida (30 avaliações), que é mais estável, e não a
acurácia de um único holdout. A ordem dos modelos pode mudar com outra `--seed`.

## 2. Módulo de inferência

```bash
# linha de comando
python inferencia.py --age 40-49 --menopause premeno --tumor-size 30-34 --inv-nodes 6-8 \
    --node-caps yes --deg-malig 3 --breast left --breast-quad left_low --irradiat yes

# modo interativo
python inferencia.py --interativo

# várias pacientes (JSON ou CSV com cabeçalho), saída opcional em JSON
python inferencia.py --arquivo exemplos/pacientes.json [--json]
```

Saída de exemplo:

```
Modelo: KNN (acurácia CV 76.3%, teste 67.2%)

Paciente 1: {'age': '40-49', 'menopause': 'premeno', 'tumor-size': '30-34', ...}
  Indicativo (Class): recurrence-events
  Probabilidade de recorrência: 76.2%
```

Uso como biblioteca:

```python
from inferencia import prever
prever({"age": "60-69", "menopause": "ge40", "tumor-size": "10-14", "inv-nodes": "0-2",
        "node-caps": "no", "deg-malig": "1", "breast": "right", "breast-quad": "left_up",
        "irradiat": "no"})
# [{'Class': 'no-recurrence-events', 'prob_recurrence_events': 0.0476, ...}]
```

Os valores são validados contra os domínios da base. Atributos não informados (ou `?`) são imputados pela moda.

| Atributo | Valores |
|---|---|
| age | 10-19, 20-29, …, 90-99 |
| menopause | lt40, ge40, premeno |
| tumor-size | 0-4, 5-9, …, 55-59 |
| inv-nodes | 0-2, 3-5, …, 36-39 |
| node-caps | yes, no |
| deg-malig | 1, 2, 3 |
| breast | left, right |
| breast-quad | left_up, left_low, right_up, right_low, central |
| irradiat | yes, no |

> Trabalho acadêmico. O resultado é apenas um indicativo e não substitui avaliação médica.
