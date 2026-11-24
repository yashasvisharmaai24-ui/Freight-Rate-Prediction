from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results.json"
OUTPUT = ROOT / "reports" / "freight_rate_model_report.pdf"
TEAL = colors.HexColor("#064A56")


def money(value: float) -> str:
    return f"${value:,.2f}"


def on_page(canvas, document) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#D9E2E4"))
    canvas.line(document.leftMargin, height - 0.55 * inch, width - document.rightMargin, height - 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#455A60"))
    canvas.drawString(document.leftMargin, 0.42 * inch, "Freight Rate Prediction | Assessment report")
    canvas.drawRightString(width - document.rightMargin, 0.42 * inch, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> None:
    if not RESULTS.is_file():
        raise FileNotFoundError("Run src/pipeline.py first to generate measured results.")
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ProjectTitle", parent=styles["Title"], textColor=TEAL, fontSize=24, leading=29, alignment=TA_LEFT, spaceAfter=12))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=TEAL, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=9, leading=13))
    story = [
        Paragraph("Freight Rate Prediction", styles["ProjectTitle"]),
        Paragraph("Machine Learning Engineer Assessment", styles["Heading2"]),
        Paragraph("Model development, time-based validation, and December 2025 scenario predictions", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Executive summary", styles["Section"]),
    ]

    for comparison in results["comparisons"]:
        selected = next(row for row in comparison["models"] if row["model"] == comparison["selected_model"])
        story.append(
            Paragraph(
                f"<b>{escape(comparison['name'].replace('_', ' ').title())}:</b> "
                f"{escape(comparison['selected_model'])} selected on chronological holdout MAE "
                f"({money(selected['mae'])}); RMSE {money(selected['rmse'])}; R-squared {selected['r2']:.4f}.",
                styles["BodyText"],
            )
        )
    story.extend(
        [
            Spacer(1, 8),
            Paragraph(
                f"The final training file has {results['data']['rows']:,} labeled shipments from "
                f"{results['data']['date_min']} through {results['data']['date_max']}. "
                f"The time-based holdout contains {results['split']['holdout_rows']:,} later shipments "
                f"starting {results['split']['holdout_date_start']}; the final validation output contains "
                f"{results['validation']['rows']:,} predictions. The December scenario uses only its "
                "seven supplied columns and does not rely on coordinates or market/quote indicators.",
                styles["BodyText"],
            ),
            Paragraph("Problem and data", styles["Section"]),
            Paragraph(
                "Predict the positive posted freight rate for each shipment. Training includes route names, "
                "coordinates, distance, equipment, weight, date, market index, quote signal, and the target. "
                "The validation file supplies all predictors but withholds the target; the December scenario "
                "supplies route, distance, equipment, weight, date, and a blank prediction field.",
                styles["BodyText"],
            ),
            Paragraph("Exploratory analysis", styles["Section"]),
            Paragraph(
                f"Rates range from {money(results['data']['target']['min'])} to "
                f"{money(results['data']['target']['max'])}; median {money(results['data']['target']['median'])}. "
                f"Distance has Pearson correlation {results['data']['numeric_target_correlations'].get('distance', float('nan')):.3f} "
                "with posted rate. Equipment and time patterns are visualized below.",
                styles["BodyText"],
            ),
            Image(str(ROOT / "reports" / "eda_overview.png"), width=7.1 * inch, height=4.75 * inch),
            PageBreak(),
            Paragraph("Data quality and feature preparation", styles["Section"]),
        ]
    )
    missing_rows = [["Field", "Missing rows"]] + [
        [field, f"{count:,}"] for field, count in results["data"]["missing_values"].items() if count
    ]
    if len(missing_rows) == 1:
        missing_rows.append(["None", "0"])
    missing_table = Table(missing_rows, colWidths=[2.5 * inch, 1.4 * inch], repeatRows=1)
    missing_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5D8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F6")]),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            Paragraph(
                f"There are {results['data']['duplicate_load_ids']} duplicate training IDs and "
                f"{results['data']['duplicate_full_rows']} fully duplicated rows. "
                f"{results['data']['negative_weights']} negative weights (and "
                f"{results['data']['nonpositive_weights']} non-positive weights) were treated as missing; "
                "weights and other numeric predictors are median-imputed, with missingness indicators. "
                "Categorical values are one-hot encoded with unseen categories ignored safely. "
                "Target outliers were retained; no target values were imputed or clipped.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
            missing_table,
            Spacer(1, 8),
            Table(
                [[
                    Image(str(ROOT / "reports" / "numeric_distributions.png"), width=3.3 * inch, height=1.98 * inch),
                    Image(str(ROOT / "reports" / "rate_by_equipment.png"), width=3.3 * inch, height=1.65 * inch),
                ]],
                colWidths=[3.45 * inch, 3.45 * inch],
                style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)],
            ),
            Paragraph("Feature sets and validation design", styles["Section"]),
            Paragraph(
                f"The training data has {results['data']['unique_pickups']} pickup cities and "
                f"{results['data']['unique_deliveries']} delivery cities. IQR screening flagged "
                f"{results['data']['iqr_outlier_counts'].get('distance', 0)} distances and "
                f"{results['data']['iqr_outlier_counts'].get('posted_rate', 0)} target rates; these are "
                "review flags, not automatic exclusions. Both feature sets derive month, day, weekday, and "
                "cyclical day-of-year values from the date. "
                "The full validation model additionally uses available coordinates, market index, and quote "
                "signal. The December model excludes those unavailable fields and uses only route names, "
                "equipment, distance, weight, and date. A chronological split assigns the earliest 80% of "
                "distinct training dates to fitting and the latest 20% to model selection, without shuffling.",
                styles["BodyText"],
            ),
            PageBreak(),
            Paragraph("Model comparison", styles["Section"]),
        ]
    )
    comparison_data = [["Feature regime", "Model", "MAE", "RMSE", "R-squared"]]
    for comparison in results["comparisons"]:
        for model in comparison["models"]:
            comparison_data.append(
                [
                    comparison["name"].replace("_", " "),
                    model["model"],
                    money(model["mae"]),
                    money(model["rmse"]),
                    f"{model['r2']:.4f}",
                ]
            )
    comparison_table = Table(comparison_data, colWidths=[1.55 * inch, 1.55 * inch, 1.1 * inch, 1.1 * inch, 1.0 * inch], repeatRows=1)
    comparison_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5D8")),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F6")]),
            ]
        )
    )
    story.extend(
        [
            comparison_table,
            Spacer(1, 10),
            Image(str(ROOT / "reports" / "model_comparison.png"), width=7.0 * inch, height=2.65 * inch),
            Paragraph(
                "Model selection is based on measured chronological holdout MAE. Final fitted artifacts use "
                "all labeled rows. Validation predictions are not labeled ground truth, so no final-validation "
                "accuracy is reported. The supplied scorer checks output format and creates the scenario chart; "
                "it does not calculate predictive accuracy.",
                styles["SmallBody"],
            ),
            PageBreak(),
            Paragraph("December 2025 scenario", styles["Section"]),
            Paragraph(
                f"Fixed shipment: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. "
                f"The selected available-feature model produces daily predictions from "
                f"{results['december']['date_min']} through {results['december']['date_max']}; "
                f"range {money(results['december']['prediction_min'])}–"
                f"{money(results['december']['prediction_max'])}, median "
                f"{money(results['december']['prediction_median'])}.",
                styles["BodyText"],
            ),
        ]
    )
    chart = ROOT / "scorer_results" / "candidate_december.png"
    if chart.is_file():
        story.append(Image(str(chart), width=7.0 * inch, height=3.12 * inch))
    else:
        story.append(Paragraph("The scorer chart has not yet been generated.", styles["BodyText"]))
    story.extend(
        [
            Paragraph("Reproducibility and limitations", styles["Section"]),
            Paragraph(
                f"Environment versions: Python {results['python_version']}, pandas "
                f"{results['pandas_version']}, NumPy {results['numpy_version']}, "
                f"scikit-learn {results['scikit_learn_version']}. Model artifacts are stored under models/. "
                "Time-based holdout results are a retrospective estimate and may not fully reflect later "
                "market shifts. The final validation target is unavailable in the supplied files, and the "
                "assessment scorer explicitly does not assess prediction quality.",
                styles["BodyText"],
            ),
        ]
    )

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.78 * inch,
        bottomMargin=0.68 * inch,
        title="Freight Rate Prediction - Assessment Report",
        author="Candidate",
    )
    document.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_report()
