# Loom walkthrough (about 2–3 minutes)

## 0:00–0:20 — Objective

“I built a freight-rate regression workflow using the supplied historical shipment data. The goal was to estimate posted rates, produce the required 12,000 validation predictions, and forecast a fixed Lexington-to-Fort Wayne shipment for every day in December 2025.”

## 0:20–0:50 — Data and quality

“The labeled training set contains 48,000 shipments dated January through October 2025. I checked duplicates, missing values, distributions, and date trends. There are 300 missing weights, 292 negative weights, and 374 missing market-index values. Since non-positive weights are not meaningful shipment weights, I treated them as missing and imputed numeric features within the training pipeline. The target rates were positive, so I retained the observed target values.”

## 0:50–1:20 — Validation design and features

“I used a chronological holdout: the earlier dates were used to fit candidate models and the latest dates, starting September 1, were held out. That is more representative of predicting future shipments than a random split. The model pipeline imputes numeric values, one-hot encodes route and equipment categories, and handles categories not seen during training. I derived calendar features from each shipment date.”

## 1:20–1:55 — Model comparison

“I compared a median baseline, Extra Trees, Random Forest, and HistGradientBoosting using MAE, RMSE, and R-squared. I selected by holdout MAE. HistGradientBoosting was best in both feature regimes. On the full-feature holdout it achieved MAE $133.82, RMSE $638.39, and R-squared 0.8250. On the December-compatible holdout it achieved MAE $138.99, RMSE $637.75, and R-squared 0.8254. These are measured holdout results, not scores on the unlabeled final validation data.”

## 1:55–2:25 — Outputs and December forecast

“The validation predictions preserve the template’s load IDs and contain the required two columns. For December, the input only provides pickup, delivery, distance, equipment, weight, and date. I trained and selected a separate model using those available features only; I did not invent route coordinates or market signals. The scorer confirmed all 12,000 validation rows and 31 December dates and generated the candidate chart.”

## 2:25–2:45 — Close

“The repository includes the training pipeline, fitted pipelines, EDA charts, actual model comparison metrics, scorer output, and a PDF report. The provided scorer checks file formatting and chart generation; final validation accuracy is assessed separately by the company.”

Before recording, show the scorer output and `scorer_results/candidate_december.png`.
