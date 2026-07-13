# Fake News Classification — ISOT Dataset

NLP capstone project comparing classical machine-learning models with a fine-tuned BERT classifier for binary news classification.

## Research question

How accurately can news articles from the ISOT dataset be classified as `FAKE` or `REAL` using article text, and how much of the performance is explained by source-specific artifacts?

This project does **not** claim to perform full fact-checking. It evaluates text classification within the ISOT dataset and explicitly examines its limitations.

## Dataset

The project uses the **ISOT Fake News Dataset**, which contains:

- `True.csv`: real news articles, predominantly from Reuters;
- `Fake.csv`: fake news articles collected from other websites;
- fields: `title`, `text`, `subject`, and `date`.

The CSV files are not included in the repository. Download both files from the official dataset page or its Kaggle mirror and place them in the project root.

## Important dataset limitation

The target label is strongly associated with source and writing style:

- most real articles come from Reuters;
- fake articles come from different websites;
- subject categories are also unevenly distributed between classes.

Therefore, a model may learn source style, editorial templates, or topic differences instead of factual truthfulness.

The project addresses this limitation by:

1. removing duplicate articles before splitting;
2. inspecting the relationship between source markers and labels;
3. training an additional Logistic Regression model after removing explicit source markers;
4. analyzing the most influential TF-IDF features;
5. describing the results as **in-domain classification**, not automated fact verification.

## Models

1. **TF-IDF + Logistic Regression**
2. **TF-IDF + calibrated Linear SVM**
3. **Fine-tuned `bert-base-uncased`**
4. **Ablation:** TF-IDF + Logistic Regression after removing explicit source markers

## Data split

A stratified split is applied to the complete DataFrame:

- 70% training;
- 15% validation;
- 15% testing.

The DataFrame is split directly so that article text, title, subject, label, and predictions remain aligned during error analysis.

## Evaluation

Because `FAKE = 0`, fake-class metrics are calculated explicitly with `pos_label=0`.

Reported metrics:

- accuracy;
- balanced accuracy;
- precision for FAKE;
- recall for FAKE;
- F1 for FAKE;
- ROC-AUC;
- Matthews Correlation Coefficient;
- training time.

The notebook also generates:

- confusion matrices;
- ROC curves;
- qualitative error examples;
- top TF-IDF features.

## Previously obtained results

The earlier project run produced approximately:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9951 | 0.9947 | 0.9950 | 0.9949 | 0.9996 |
| Fine-tuned BERT | 0.9997 | 1.0000 | 0.9994 | 0.9997 | 1.0000 |
| Logistic Regression with explicit source markers removed | 0.9892 | 0.9873 | 0.9900 | 0.9887 | 0.9989 |

These values should be interpreted cautiously. Near-perfect scores are evidence that the ISOT split is easy for modern text models, not proof that the models solve real-world fake-news detection.

The revised notebook recalculates all results after duplicate removal and corrected error-analysis logic, so exact values may differ from the earlier run.

## Key findings

- Classical TF-IDF models can achieve extremely high in-domain performance at low computational cost.
- Removing explicit source markers reduces performance, indicating that source-related artifacts contribute to the result.
- A small absolute reduction in F1 can still represent a substantial increase in the number of errors.
- Very high BERT performance on a random split does not guarantee generalization to unseen publishers or later time periods.
- The most defensible conclusion is that the models classify ISOT articles well, while external validation is still required.

## Repository structure

```text
.
├── fake_news_classification_reworked.ipynb
├── README.md
├── requirements.txt
├── True.csv                         # download separately
├── Fake.csv                         # download separately
└── generated after execution:
    ├── final_results.csv
    ├── eda_plots.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── top_features.png
    └── models/bert_fake_news/
```

## Running the project

### Google Colab

1. Open the notebook in Colab.
2. Select a GPU runtime for BERT training.
3. Upload `True.csv` and `Fake.csv`.
4. Run all cells in order.

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook fake_news_classification_reworked.ipynb
```

On Windows:

```bash
.venv\Scripts\activate
```

## Limitations

- random splitting allows the same publishers and writing styles to appear in train and test;
- explicit marker removal does not remove all stylistic differences;
- the model does not retrieve evidence or verify claims;
- BERT truncates long articles;
- the dataset focuses largely on political news;
- external generalization is not established.

## Recommended future work

- source-held-out evaluation;
- chronological train/test split;
- testing on LIAR, WELFake, or FakeNewsNet;
- near-duplicate detection using semantic similarity;
- robustness testing with paraphrased articles;
- retrieval-augmented claim verification;
- calibration analysis and threshold tuning;
- comparison of speed, memory, and inference latency.

## Conclusion

The project demonstrates a complete NLP development cycle: data analysis, preprocessing, duplicate handling, model training, evaluation, interpretability, leakage analysis, and discussion of limitations.

Its central result is not that fake news has been “solved,” but that strong benchmark scores can be produced by dataset-specific patterns. Recognizing that distinction is essential for building trustworthy NLP systems.

## References

1. Ahmed, H., Traore, I., & Saad, S. (2018). Detection of online fake news using N-gram analysis and machine learning techniques.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
3. ISOT Fake News Dataset, University of Victoria.
