# Freight Rate Prediction Challenge

Solution to the Machine Learning Engineer assessment, following the provided
submission instructions.

## What to do

1. Train and validate using `data/train_test.csv`.
2. Predict every row in `data/validation.csv`.
3. Match predictions to `data/validation_predictions_template.csv` and save
   `validation_predictions.csv`.
4. Fill `predicted_rate` in `data/december_chart_inputs.csv`.
5. Install dependencies and run the supplied scorer:

   ```powershell
   python -m pip install -r requirements.txt
   python score.py --predictions validation_predictions.csv --december-predictions data\december_chart_inputs.csv
   ```

The scorer checks output formatting and creates
`scorer_results/candidate_december.png`; it does not calculate model accuracy.

## Project layout

```text
data/                       Supplied CSV inputs and completed December scenario
models/                     Serialized fitted scikit-learn pipelines
reports/                    EDA charts, measured results, and PDF report
scorer_results/             Official December chart
src/pipeline.py             EDA, model comparison, fitting, and predictions
src/create_report.py        PDF report generation
score.py                    Assessment formatting checks and chart generator
validation_predictions.csv  Final 12,000-row submission file
```

## Setup

Python 3.10+ is recommended. In PowerShell, from the project directory, create
and activate an isolated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project was also run successfully with Python 3.14.7.

## Train and generate predictions

```powershell
python src\pipeline.py
```

This reads the official input paths under `data/`, performs data checks and
exploratory analysis, evaluates a median baseline, Extra Trees, Random Forest,
and HistGradientBoosting on a chronological holdout, selects by holdout MAE,
refits on all labeled records, and creates:

- `validation_predictions.csv` (`load_id,predicted_rate`, in template order)
- `data/december_chart_inputs.csv` (completed seven-column scenario file)
- `models/validation_full_features.joblib`
- `models/december_available_features.joblib`
- `reports/results.json` and EDA/model-comparison plots

Non-positive weights are treated as missing and imputed within the fitted
preprocessing pipeline. Unseen route categories are supported. The December
model uses only pickup, delivery, distance, equipment, weight, and date; it
does not invent coordinates or market/quote data.

## Run the assessment scorer and create the report

```powershell
python score.py --predictions validation_predictions.csv --december-predictions data\december_chart_inputs.csv
python src\create_report.py
```

Outputs include `scorer_results/candidate_december.png` and
`reports/freight_rate_model_report.pdf`. The report includes the validation
strategy, measured model comparisons, EDA, and the scorer-generated December
chart.

## Evaluation

The script writes measured MAE, RMSE, and R-squared values for every compared
model and both feature regimes to `reports/results.json`. Metrics in the
report come from that file; no score is claimed for the unlabeled final
validation set.

## Submission

Submit the repository, `validation_predictions.csv`, the PDF report, and your
2–3 minute Loom link. The assessment requires camera-on video covering key
data findings, data-quality treatment, model choice, the split/validation
approach, and a short code walkthrough. Add the official PDF to the repository
if you have a copy of the file itself.
