# Domain-Specific PCA for Embedding Dimensionality Reduction

Im Rahmen eines Experiments untersucht dieses Repository, ob **domänenspezifisch trainierte PCA-Modelle** die Retrieval-Qualität komprimierter Embedding-Vektoren im Vergleich zu einer **generischen PCA-Baseline** verbessern können. Die Analyse deckt drei Fachdomänen ab: **Medizin**, **Recht** und **Finanzwesen**.

## Überblick

Die Pipeline besteht aus drei aufeinander aufbauenden Schritten:

1. **Embedding-Erzeugung** – Texte aus Benchmark-Datensätzen werden mit dem Modell [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) in 384-dimensionale Vektoren umgewandelt.
2. **PCA-Training** – Für jede Domäne wird eine PCA-Pipeline (StandardScaler + PCA) auf einem 60 %-Trainings-Split trainiert. Zusätzlich wird eine generische Pipeline auf dem NQ-Datensatz trainiert.
3. **Evaluation** – Die komprimierten Vektoren werden mittels FAISS-basiertem Retrieval über alle Dimensionsstufen (1–384) evaluiert und anhand von Retrieval-Metriken (MRR@10, NDCG@10, Precision@1, Hit Rate@10, Mean Rank, Latenz) verglichen.

## Projektstruktur

```
domain-pca/
├── src/                              # Jupyter Notebooks (Pipeline)
│   ├── 01_generate_embeddings.ipynb  # Embedding-Erzeugung für alle Domänen
│   ├── 02_train_pca.ipynb            # PCA-Training (generisch + domänenspezifisch)
│   └── 03_evaluation.ipynb           # FAISS-Retrieval & Metrik-Auswertung
├── models/                           # Trainierte PCA-Pipelines (.joblib)
│   ├── pca_pipeline_generic.joblib
│   ├── pca_pipeline_medicine.joblib
│   ├── pca_pipeline_law.joblib
│   └── pca_pipeline_finance.joblib
├── results/                          # Evaluationsergebnisse pro Domäne
│   ├── finance/
│   │   ├── generic/                  # Ergebnisse mit generischer PCA
│   │   └── specific/                 # Ergebnisse mit domänenspezifischer PCA
│   ├── generic/
│   │   └── generic/
│   ├── law/
│   │   ├── generic/
│   │   └── specific/
│   └── medicine/
│       ├── generic/
│       └── specific/
├── data/                             # NICHT IM REPOSITORY ENTHALTEN (siehe Hinweis)
├── environment.yml                   # Conda-Umgebung mit allen Abhängigkeiten
└── README.md
```

## Hinweis zum `data/`-Ordner

Der Ordner `data/` wurde **nicht hochgeladen**, da er zu groß ist. Er enthält die generierten Embedding-Vektoren (`.npy`-Dateien), IDs, Ground-Truth-Zuordnungen (QRels) sowie die Trainings-/Evaluations-Splits für alle Domänen.


### Erwartete Struktur des `data/`-Ordners

```
data/
├── generic/          # Natural Questions (NQ)
│   ├── corpus/       # Corpus-Embeddings (.npy)
│   ├── queries/      # Query-Embeddings (.npy)
│   ├── ids/          # Dokument-/Query-IDs + QRels (.npy, .json)
│   └── splits/       # Train/Eval-Split (60/40)
├── medicine/         # PubMedQA
│   ├── corpus/
│   ├── queries/
│   ├── ids/
│   └── splits/
├── law/              # LegalBenchRAG
│   ├── corpus/
│   ├── queries/
│   ├── ids/
│   └── splits/
└── finance/          # FinanceBench
    ├── corpus/
    ├── queries/
    ├── ids/
    └── splits/
```

## Verwendete Benchmarks

| Domäne | Datensatz | Corpus-Größe | Queries |
|--------|-----------|-------------|---------|
| Generisch | [Natural Questions (NQ)](https://huggingface.co/datasets/beir/nq) | 100.000 | 2.924 |
| Medizin | [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | 1.000 | 1.000 |
| Recht | [LegalBenchRAG](https://github.com/zeroentropy-ai/legalbenchrag) | 698 | 6.889 |
| Finanzwesen | [FinanceBench](https://github.com/patronus-ai/financebench) | 150 | 150 |

## Evaluationsmetriken

- **MRR@10** – Mean Reciprocal Rank der Top-10-Ergebnisse
- **NDCG@10** – Normalized Discounted Cumulative Gain
- **Precision@1** – Anteil korrekter Top-1-Treffer
- **Hit Rate@10** – Anteil der Queries mit mindestens einem Treffer in den Top-10
- **Mean Rank** – Durchschnittlicher Rang des ersten relevanten Treffers
- **Latenz** – Suchzeit pro Query (ms), gemessen via FAISS IndexFlatL2

## Setup


### Installation

Alle benötigten Pakete und Abhängigkeiten sind in der Datei `environment.yml` definiert.

### Ausführung

Die Notebooks müssen in der folgenden Reihenfolge ausgeführt werden:

```
1. src/01_generate_embeddings.ipynb   # Embeddings erzeugen
2. src/02_train_pca.ipynb             # PCA-Modelle trainieren
3. src/03_evaluation.ipynb            # Evaluation durchführen
```

## Technologie-Stack

- **Embedding-Modell:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 Dimensionen)
- **Dimensionsreduktion:** scikit-learn PCA + StandardScaler
- **Vektorsuche:** FAISS (IndexFlatL2)
- **Datenquellen:** Hugging Face Datasets, lokale Dateien
- **Visualisierung:** Matplotlib, Seaborn, Plotly
