[README.md](https://github.com/user-attachments/files/35118628/README.md)
# Assignment 4 — Reweighing: Paper Review and Algorithm Evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/reweighing-fairness/blob/main/reweighing_aif360.ipynb)

This repository evaluates **Reweighing**, a pre-processing bias-mitigation algorithm, on three standard fairness datasets (Adult, German Credit and COMPAS), using the implementation in IBM's [AI Fairness 360 (AIF360)](https://github.com/Trusted-AI/AIF360) library.

**Paper:** F. Kamiran and T. Calders, "Data preprocessing techniques for classification without discrimination", *Knowledge and Information Systems* 33(1), 1–33, 2012\. [https\://doi.org/10.1007/s10115-011-0463-8](https://doi.org/10.1007/s10115-011-0463-8)

## The algorithm

Reweighing gives every training instance a weight based on its group *s* and label *c*:

```
W(s, c) = P(S = s) × P(C = c) / P(S = s, C = c)
```

Under-represented group–label combinations get weights above 1 and over-represented ones below 1, so the protected attribute and the label become statistically independent in the weighted training data. No labels or features are changed. A standard classifier is then trained with these sample weights.

## Datasets

| Dataset | Rows used | Protected attribute (privileged group) | Favourable outcome |
| :---- | :---- | :---- | :---- |
| Adult (UCI training file) | 30,162 | sex (male) | income \> 50K |
| Statlog German Credit | 1,000 | age (over 25\) | good credit risk |
| COMPAS (ProPublica two-year data) | 5,278 | race (Caucasian vs. African-American) | no reoffence within two years |

The data files are **not** included in this repository. Download them and put them in the `data/` folder (see [`data/README.md`](http://data/README.md)):

- `adult.csv`: UCI Adult, via Kaggle: [https\://www\.kaggle.com/datasets/uciml/adult-census-income](https://www.kaggle.com/datasets/uciml/adult-census-income)  
- `german.data`: UCI Statlog German Credit: [https\://archive.ics.uci.edu/dataset/144](https://archive.ics.uci.edu/dataset/144)  
- `propublica_data_for_fairml.csv`: COMPAS (FairML subset of ProPublica's data), via Kaggle: [https\://www\.kaggle.com/danofer/compass](https://www.kaggle.com/danofer/compass)

## Pipeline

1. Compute **DP\_Diff** (statistical parity difference), **DIR** (disparate impact ratio) and **consistency** on the original labels.  
2. Split the data 70/30, stratified on label × protected group; repeat for 5 random seeds (0–4).  
3. Fit AIF360's `Reweighing` on the training split only. The notebook asserts that its weights equal the paper's formula.  
4. Train `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)` with the weights. An unweighted model is the baseline. Following the paper's main setting (assumption A2), the protected attribute is used for the weights and metrics but is **not** a classifier input.  
5. Evaluate DP\_Diff, DIR, consistency (k \= 5), accuracy, balanced accuracy and F1 on the test split, using AIF360's `BinaryLabelDatasetMetric` and `ClassificationMetric`.  
6. Reference run: the paper's second setting, where the classifier may also use the protected attribute.

## Results (test set, mean over 5 splits)

| Dataset | Model | DP\_Diff (ideal 0\) | DIR (ideal 1\) | Consistency | Accuracy | Balanced acc. |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Adult | Baseline | −0.189 | 0.30 | 0.958 | 0.848 | 0.769 |
| Adult | Reweighing | −0.099 | 0.56 | 0.959 | 0.842 | 0.752 |
| German Credit | Baseline | −0.114 | 0.85 | 0.814 | 0.745 | 0.673 |
| German Credit | Reweighing | −0.076 | 0.90 | 0.813 | 0.738 | 0.661 |
| COMPAS | Baseline | −0.238 | 0.68 | 0.992 | 0.665 | 0.661 |
| COMPAS | Reweighing | −0.238 | 0.68 | 0.992 | 0.666 | 0.661 |

Original labels (before any training): DP\_Diff −0.200 / −0.149 / −0.132 and DIR 0.36 / 0.79 / 0.78 for Adult / German Credit / COMPAS.

**Summary:**

- Reweighing roughly halves the parity gap on Adult and reduces it modestly on German Credit, where the gain is smaller than the variation between splits.  
- It leaves COMPAS unchanged: without race as an input, a linear model on COMPAS's five features barely reacts to the weights.  
- Accuracy drops by less than one point.  
- When the classifier may use the protected attribute (reference setting), DP\_Diff after reweighing is −0.095 (Adult), −0.045 (German Credit) and −0.031 (COMPAS).

Full tables are in [`results/`](http://results/), and the chart is in `results/fairness_comparison.png`.

## How to run

**Google Colab:** click the badge above, then choose *Runtime → Run all*. The notebook installs AIF360 and asks you to upload the three data files. You can upload the zip archives instead; the notebook extracts them.

**Locally (Python 3.10+):**

```sh
pip install -r requirements.txt
# put adult.csv, german.data and propublica_data_for_fairml.csv in data/
jupyter notebook reweighing_aif360.ipynb
```

The notebook already contains the outputs of a complete run, so the results can be read without running it.

## Repository structure

```
reweighing-fairness/
├── reweighing_aif360.ipynb   # full pipeline, with saved outputs
├── requirements.txt
├── data/                     # put the three data files here (not committed)
├── results/                  # CSV result tables and the comparison chart
└── docs/report.pdf           # paper review, evaluation and highlighted paper
```

## References

- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems, 33*(1), 1–33. [https\://doi.org/10.1007/s10115-011-0463-8](https://doi.org/10.1007/s10115-011-0463-8)  
- Calders, T., Kamiran, F., & Pechenizkiy, M. (2009). Building classifiers with independency constraints. *IEEE International Conference on Data Mining Workshops (ICDMW 2009\)*.  
- Bellamy, R. K. E., et al. (2018). AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias. arXiv:1810.01943. [https\://github.com/Trusted-AI/AIF360](https://github.com/Trusted-AI/AIF360)  
- Becker, B., & Kohavi, R. (1996). Adult \[Dataset\]. UCI Machine Learning Repository. [https\://doi.org/10.24432/C5XW20](https://doi.org/10.24432/C5XW20)  
- Hofmann, H. (1994). Statlog (German Credit Data) \[Dataset\]. UCI Machine Learning Repository. [https\://doi.org/10.24432/C5NC77](https://doi.org/10.24432/C5NC77)  
- Larson, J., Mattu, S., Kirchner, L., & Angwin, J. (2016). How we analyzed the COMPAS recidivism algorithm. ProPublica. [https\://www\.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)
