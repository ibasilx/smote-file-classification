# File Classification Using SMOTE and Machine Learning

An academic / portfolio machine-learning project that evaluates five classifiers on a 20-class file-type feature dataset, with experiments before and after SMOTE and a chi-squared feature-selection variant.

**Live project showcase:** [Explore the project website](https://ibasilx.github.io/smote-file-classification/).

## Overview

The work studies the effect of class imbalance on multiclass classification metrics. It is a research and portfolio project with cybersecurity relevance through automated file categorization. It is **not** a malware detector, antivirus, or production security tool.

## Dataset

The supplied dataset is named sift_512.csv and has 47,482 samples, 512 input columns named F0–F511, and a Class target column. The 20 integer class labels map to file-type names in the original analysis code. The filename does not verify how the features were generated: the feature-extraction process is not included, so this repository does not claim that the columns are confirmed SIFT descriptors.

The CSV is about 80.5 MB and is not committed. The source and redistribution license are not documented in the supplied files. See data/README.md for the expected local dataset layout.

## Methodology

- Split the data into 80% training and 20% testing with stratification and random_state=42.
- Train and evaluate Random Forest, SVC (RBF), Decision Tree, AdaBoost, and MLP.
- Run a baseline experiment on the original training split.
- Run a second experiment after applying SMOTE to the training split only. The test data is never resampled.
- Run the feature-selection experiment after SMOTE: fit MinMaxScaler on the resampled training matrix, then use SelectKBest(chi2, k=256). Apply the fitted scaler and selector to the test split.
- Report accuracy, macro precision, macro recall (true-positive rate), macro F1, and macro false-positive rate.

Baseline and SMOTE experiments use 512 features. The feature-selection experiment uses 256 selected columns. It selects features; it does not perform PCA or another projection.

## Results

The following values are copied from the supplied result files. Percentages are shown as reported; precision, recall, and F1 are macro-averaged in the original evaluation code.

| Experiment | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Baseline | Random Forest | 57.3444% | 46.6252% |
| Baseline | SVC | 52.8483% | 38.3663% |
| Baseline | Decision Tree | 44.7931% | 37.1469% |
| Baseline | AdaBoost | 17.4687% | 6.9723% |
| Baseline | MLP | 39.3071% | 32.6869% |
| SMOTE | Random Forest | 56.9969% | 54.5644% |
| SMOTE | SVC | 52.1849% | 42.9600% |
| SMOTE | Decision Tree | 40.8339% | 34.7013% |
| SMOTE | AdaBoost | 15.7629% | 9.4755% |
| SMOTE | MLP | 40.4233% | 35.6032% |
| SMOTE + feature selection | Random Forest | 54.7963% | 52.2440% |
| SMOTE + feature selection | SVC | 50.3738% | 41.8117% |
| SMOTE + feature selection | Decision Tree | 39.1703% | 32.6979% |
| SMOTE + feature selection | AdaBoost | 14.4151% | 8.5607% |
| SMOTE + feature selection | MLP | 48.1310% | 44.4510% |

The complete recorded metric table, including precision, recall, and false-positive rate, is in results/metrics.csv. These are results from the supplied run; exact matches from retraining can depend on library versions and environment.

## Local dashboard

The optional Streamlit dashboard compares the recorded results in results/metrics.csv:

~~~bash
python -m pip install -r requirements.txt
streamlit run dashboard.py
~~~

It visualizes model metrics only. It does not upload or classify raw files.

## Training

Place your authorized copy of the dataset at data/sift_512.csv, then run:

~~~bash
python -m pip install -r requirements.txt
python src/train.py
~~~

Optional paths:

~~~bash
python src/train.py --data path/to/sift_512.csv --results-dir results/generated --plots-dir plots/generated
~~~

The script writes per-experiment text reports, a generated metrics CSV, and class-distribution plots. The supplied repository does not include a fitted model export or prediction-only function.

## Project structure

~~~text
.
├── .gitignore
├── README.md
├── requirements.txt
├── dashboard.py
├── data/
│   └── README.md
├── docs/
│   └── index.html
├── results/
│   └── metrics.csv
└── src/
    └── train.py
~~~

## Limitations

- The dataset's feature-generation process is not included or verified.
- There is no raw-file upload interface and arbitrary files cannot be classified by this repository.
- No trained estimator, fitted scaler, or fitted feature selector is saved for later inference.
- The dataset source and redistribution license were not supplied; confirm permission before sharing it.
- The results are from an academic experiment and do not establish real-world security performance.

## Cybersecurity relevance

File categorization and automated file analysis can be useful components in broader security workflows. This project only evaluates file-type classification from a precomputed feature table; it does not analyze maliciousness, detect threats, or replace security controls.

## Technologies

Python, pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, and Streamlit.
