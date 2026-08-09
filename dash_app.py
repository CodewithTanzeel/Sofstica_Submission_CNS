"""
Dash / Plotly front-end for the DNS Exfiltration Detection pipeline.

Tabs:
  1. Overview        - KPI cards + model vs. baseline metrics + confusion matrix
  2. Score traffic    - upload new stateful+stateless CSVs OR pick an existing captured
                        session, get a live score, download results as CSV
  3. Model internals  - feature importances + latency, from results/inference_bundle.pkl

Run locally:
    python dash_app.py
Then open http://127.0.0.1:8050

Deploy: see the bottom of this file / the accompanying Procfile.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, dash_table
from dash import Dash
from src.score_new import load_bundle, score_dataframe


  
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LABEL_NAMES = {0: "BENIGN", 1: "LIGHT ATTACK", 2: "HEAVY ATTACK"}

MODEL_COLOR = "#EF476F"
BASELINE_COLOR = "#8D99AE"
BENIGN_COLOR = "#06D6A0"
ACCENT = "#118AB2"
BG = "#0B132B"

# ---------------------------------------------------------------------------
# Load everything once at startup
# ---------------------------------------------------------------------------
scoring = json.loads((RESULTS / "scoring_output.json").read_text(encoding="utf-8"))
predictions = pd.read_csv(RESULTS / "predictions.csv")
bundle = load_bundle(RESULTS / "inference_bundle.pkl")
threshold = bundle["threshold"]
model = bundle["model"]

model_metrics = scoring["model_test_metrics"]["per_class"]["combined"]
baseline_metrics = scoring["baseline_test_metrics"]["per_class"]["combined"]

app = dash.Dash(
    __name__,
    title="DNS Exfiltration Detector",
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.config.suppress_callback_exceptions = True


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(fig: go.Figure, **kw) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font_color="#E0E6ED",
        margin=dict(l=10, r=10, t=40, b=10),
        **kw,
    )
    return fig


def metrics_bar_figure() -> go.Figure:
    rows = []
    for label, m in [("Baseline", baseline_metrics), ("Model", model_metrics)]:
        rows.append({"Approach": label, "PR-AUC": m["pr_auc"], "Precision": m["precision"],
                      "Recall": m["recall"], "FPR": m["fpr"]})
    df = pd.DataFrame(rows).melt(id_vars="Approach", var_name="Metric", value_name="Value")
    fig = px.bar(df, x="Metric", y="Value", color="Approach", barmode="group",
                 color_discrete_map={"Baseline": BASELINE_COLOR, "Model": MODEL_COLOR},
                 title="Model vs. non-ML baseline (test set)")
    return _style(fig, yaxis_range=[0, 1])


def per_class_figure() -> go.Figure:
    rows = [{"Class": c.upper(), "PR-AUC": scoring["model_test_metrics"]["per_class"][c]["pr_auc"]}
            for c in ("light", "heavy")]
    fig = px.bar(pd.DataFrame(rows), x="Class", y="PR-AUC", color="Class", title="Model PR-AUC by attack intensity",
                 color_discrete_sequence=[MODEL_COLOR, ACCENT])
    return _style(fig, yaxis_range=[0, 1], showlegend=False)


def confusion_matrix_figure() -> go.Figure:
    cm = scoring["model_test_metrics"]["confusion_matrix"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                     labels=dict(x="Predicted", y="Actual", color="Count"),
                     x=["Benign", "Attack"], y=["Benign", "Attack"],
                     title="Model confusion matrix (test set)")
    return _style(fig, coloraxis_showscale=False)


def gauge_figure(score: float, thr: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": ".3f", "font": {"color": "#E0E6ED"}},
        title={"text": "Model score", "font": {"color": "#E0E6ED"}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#E0E6ED"},
            "bar": {"color": MODEL_COLOR if score >= thr else BENIGN_COLOR},
            "threshold": {"line": {"color": "white", "width": 3}, "value": thr},
            "steps": [{"range": [0, thr], "color": "rgba(6,214,160,0.15)"},
                      {"range": [thr, 1], "color": "rgba(239,71,111,0.15)"}],
            "bgcolor": "rgba(0,0,0,0)",
        },
    ))
    return _style(fig, height=260)


def importance_figure() -> go.Figure:
    ranked = model.explain(row=None)
    df = pd.DataFrame(ranked, columns=["feature", "importance"]).sort_values("importance")
    fig = px.bar(df, x="importance", y="feature", orientation="h",
                 title="Top model feature importances", color_discrete_sequence=[MODEL_COLOR])
    return _style(fig)


def parse_upload(contents: str) -> pd.DataFrame:
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))


def kpi_card(title: str, value: str, delta: str | None = None, color: str = ACCENT) -> dbc.Card:
    children = [html.Div(title, className="text-muted small text-uppercase"),
                html.H3(value, className="mb-0", style={"color": color})]
    if delta:
        children.append(html.Div(delta, className="text-muted small"))
    return dbc.Card(dbc.CardBody(children), className="shadow-sm", style={"backgroundColor": "#141E30"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
sample_options = [{"label": f, "value": f} for f in sorted(predictions["source_file"].unique())]

fpr_drop = baseline_metrics["fpr"] - model_metrics["fpr"]

overview_tab = dbc.Container([
    dbc.Row([
        dbc.Col(kpi_card("PR-AUC (model)", f"{model_metrics['pr_auc']:.2f}",
                         f"vs {baseline_metrics['pr_auc']:.2f} baseline", MODEL_COLOR), md=3),
        dbc.Col(kpi_card("Recall", f"{model_metrics['recall']:.1%}", "attacks caught", BENIGN_COLOR), md=3),
        dbc.Col(kpi_card("False positive rate", f"{model_metrics['fpr']:.1%}",
                         f"↓ {fpr_drop:.1%} vs baseline", ACCENT), md=3),
        dbc.Col(kpi_card("Median latency", f"{scoring.get('latency', {}).get('median_latency_ms_per_row', 0):.2f} ms",
                         "per row, inference only", "#E0E6ED"), md=3),
    ], className="g-3 my-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=metrics_bar_figure()), md=7),
        dbc.Col(dcc.Graph(figure=confusion_matrix_figure()), md=5),
    ], className="g-3"),
    dbc.Row([dbc.Col(dcc.Graph(figure=per_class_figure()), md=12)], className="g-3"),
    dbc.Alert(
        "Light-attack detection is meaningfully weaker than heavy-attack detection — "
        "a known, documented limitation. See reports/technical_report.md.",
        color="warning", className="mt-2",
    ),
], fluid=True)

score_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H5("Option A — pick an already-captured session", className="mt-3"),
            dcc.Dropdown(id="sample-picker", options=sample_options,
                        placeholder="Choose a captured session...", style={"color": "#000"}),

            html.H5("Option B — upload new raw feature CSVs", className="mt-4"),
            html.P("Paired stateful_features-*.csv and stateless_features-*.csv for a session.",
                   className="text-muted small"),
            dcc.Upload(id="upload-stateful", children=html.Div(["📁 Drop stateful_features CSV or click"]),
                      className="upload-box mb-2"),
            dcc.Upload(id="upload-stateless", children=html.Div(["📁 Drop stateless_features CSV or click"]),
                      className="upload-box"),
            dbc.Button("Score this session", id="score-btn", n_clicks=0, color="danger", className="mt-3"),
        ], md=4),
        dbc.Col([
            dcc.Loading(html.Div(id="score-output"), type="circle", color=MODEL_COLOR),
        ], md=8),
    ], className="g-4 my-2"),
], fluid=True)

model_tab = dbc.Container([
    dbc.Row([dbc.Col(dcc.Graph(figure=importance_figure()), md=12)], className="g-3 my-3"),
    dbc.Row([
        dbc.Col(kpi_card("Model type", model._model_name, color=ACCENT), md=4),
        dbc.Col(kpi_card("Input features", str(len(model._feature_columns)), color=ACCENT), md=4),
        dbc.Col(kpi_card("p95 latency", f"{scoring.get('latency', {}).get('p95_latency_ms_per_row', 0):.2f} ms",
                         color=ACCENT), md=4),
    ], className="g-3"),
], fluid=True)

app.layout = dbc.Container([
    dcc.Store(id="scored-data-store"),
    dcc.Download(id="download-scored-csv"),
    dbc.NavbarSimple(
        brand="DNS Exfiltration Detector",
        brand_style={"fontWeight": "bold"},
        color="dark", dark=True, fluid=True, className="mb-3",
    ),
    dbc.Tabs(
        [
            dbc.Tab(overview_tab, label="Overview", tab_id="overview-tab"),
            dbc.Tab(score_tab, label="Score traffic", tab_id="score-tab"),
            dbc.Tab(model_tab, label="Model internals", tab_id="model-tab"),
        ],
        id="tabs",
        active_tab="overview-tab",
    ),
    html.Footer(
        "Advisory tool only — outputs never block traffic or take autonomous action.",
        className="text-muted small text-center my-4",
    ),
], fluid=True, style={"backgroundColor": BG, "minHeight": "100vh", "paddingBottom": "20px"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("score-output", "children"),
    Output("scored-data-store", "data"),
    Input("score-btn", "n_clicks"),
    State("sample-picker", "value"),
    State("upload-stateful", "contents"),
    State("upload-stateless", "contents"),
    prevent_initial_call=True,
)
def run_scoring(n_clicks, sample_file, stateful_contents, stateless_contents):
    try:
        if stateful_contents and stateless_contents:
            scored = score_dataframe(bundle, parse_upload(stateful_contents), parse_upload(stateless_contents))
            source_label = "uploaded CSVs"
        elif sample_file:
            scored = predictions[predictions["source_file"] == sample_file].copy()
            if scored.empty:
                return dbc.Alert("No rows found for that session.", color="danger"), None
            source_label = sample_file
        else:
            return dbc.Alert("Pick a session from the dropdown, or upload both CSVs, then click Score.",
                             color="secondary"), None

        avg_score = scored["model_score"].mean()
        alert_rate = scored["predicted_label"].mean()
        reasons = scored[scored["predicted_label"] == 1]["top_reason_codes"].dropna()
        top_reason = reasons.iloc[0] if len(reasons) else "no alerts raised"

        out = html.Div([
            dcc.Graph(figure=gauge_figure(avg_score, threshold)),
            html.P([f"Scored {len(scored)} rows from ", html.B(source_label),
                   f" • {alert_rate:.1%} flagged • example reason codes: {top_reason}"]),
            dbc.Alert("Advisory only — this does not block traffic or take any autonomous action.",
                      color="warning"),
            dbc.Button("Download scored rows (CSV)", id="download-btn", color="secondary", size="sm",
                      className="mb-2"),
            dash_table.DataTable(
                data=scored[["model_score", "predicted_label", "top_reason_codes"]].head(20).to_dict("records"),
                columns=[{"name": c, "id": c} for c in ["model_score", "predicted_label", "top_reason_codes"]],
                style_cell={"backgroundColor": "#141E30", "color": "#E0E6ED", "fontSize": "13px"},
                style_header={"backgroundColor": "#0B132B", "fontWeight": "bold"},
                page_size=10,
            ),
        ])
        return out, scored.to_json(date_format="iso", orient="split")

    except Exception as exc:
        return dbc.Alert(f"Scoring failed: {exc}", color="danger"), None


@app.callback(
    Output("download-scored-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("scored-data-store", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, stored_json):
    if not stored_json:
        return None
    df = pd.read_json(io.StringIO(stored_json), orient="split")
    return dcc.send_data_frame(df.to_csv, "scored_results.csv", index=False)


# Minimal inline CSS for the upload boxes (kept in-file to avoid an assets/ dependency)
app.index_string = app.index_string.replace(
    "</head>",
    """
    <style>
      .upload-box { border: 1px dashed #4A5568; border-radius: 8px; padding: 14px;
                    text-align: center; color: #A0AEC0; cursor: pointer; }
      .upload-box:hover { border-color: #EF476F; color: #E0E6ED; }
    </style>
    </head>""",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)