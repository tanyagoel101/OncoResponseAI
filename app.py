from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="OncoResponse AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

LOCAL_EXPRESSION_PATH = APP_DIR / "gene_expression.csv"
LOCAL_RESPONSE_PATH = APP_DIR / "drug_response.csv"

DEFAULT_DATA_URLS = {
    "expression": [
        "https://raw.githubusercontent.com/mdozmorov/E-MTAB-3610/main/E-MTAB-3610_matrix.csv.gz",
    ],
    "annotations": [
        "https://raw.githubusercontent.com/mdozmorov/E-MTAB-3610/main/E-MTAB-3610_cell_annotations.csv.gz",
    ],
    "response": [
        "https://raw.githubusercontent.com/weiba/NIHGCN/master/Data/GDSC/cell_drug.csv",
        "https://raw.githubusercontent.com/CSB5/CaDRReS-Sc/master/preprocessed_data/GDSC/response.csv",
    ],
}

COLUMN_CANDIDATES = {
    "cell_line": [
        "CELL_LINE_NAME",
        "CellLine",
        "cell_line",
        "cell line",
        "CELL_LINE",
        "MODEL_NAME",
        "SANGER_MODEL_ID",
    ],
    "cosmic_id": [
        "COSMIC_ID",
        "COSMICID",
        "cosmic_id",
        "cell_id",
        "sample_id",
        "model_id",
    ],
    "drug": [
        "DRUG_NAME",
        "drug_name",
        "drug",
        "Drug",
        "DRUG_ID",
        "drug_id",
        "Compound",
    ],
    "cancer_type": [
        "CANCER_TYPE",
        "cancer_type",
        "TISSUE",
        "tissue",
        "TISSUE_TYPE",
        "tissue_type",
        "histology",
        "Histology",
        "primary_tissue",
    ],
    "target": [
        "IC50",
        "LN_IC50",
        "logIC50",
        "LogIC50",
        "AUC_IC50",
    ],
}

EXPECTED_STRUCTURE_MESSAGE = """
Expected files:

- `gene_expression.csv`: rows as cell lines, columns as genes, plus a cell line identifier such as `CELL_LINE_NAME` or `COSMIC_ID`
- `drug_response.csv`: one row per cell line/drug pair, with a cell line identifier, drug name, cancer or tissue type, and `IC50` or `LN_IC50`

The app first looks for local files in the project folder, then tries the built-in online sources.
"""


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(16,185,129,0.16));
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.25);
        }
        .metric-card {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 16px;
            padding: 0.8rem;
        }
        .section-label {
            font-size: 0.95rem;
            color: #93c5fd;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {normalize_token(column): column for column in columns}
    for candidate in candidates:
        candidate_key = normalize_token(candidate)
        if candidate_key in normalized:
            return normalized[candidate_key]

    for column in columns:
        column_key = normalize_token(column)
        if any(normalize_token(candidate) in column_key for candidate in candidates):
            return column
    return None


def download_to_cache(urls: list[str], destination: Path) -> Path | None:
    if destination.exists():
        return destination

    for url in urls:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            destination.write_bytes(response.content)
            return destination
        except requests.RequestException:
            continue
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def parse_annotation_identifiers(annotations_df: pd.DataFrame) -> pd.DataFrame:
    annotation_copy = annotations_df.copy()
    source_column = detect_column(annotation_copy.columns.tolist(), ["Source Name", "source_name"])
    cell_line_column = detect_column(annotation_copy.columns.tolist(), ["Characteristics[cell line]", "cell line", "CELL_LINE_NAME"])
    tissue_column = detect_column(
        annotation_copy.columns.tolist(),
        ["Characteristics[organism part]", "organism part", "tissue", "cancer_type"],
    )

    if source_column and "COSMIC_ID" not in annotation_copy.columns:
        annotation_copy["COSMIC_ID"] = (
            annotation_copy[source_column].astype(str).str.extract(r"(\d{5,})", expand=False)
        )

    if source_column and not tissue_column:
        extracted_tissue = annotation_copy[source_column].astype(str).str.extract(
            r"^[^_]+_[^_]+_[^_]+_([^_]+)_\d+$",
            expand=False,
        )
        annotation_copy["TISSUE"] = extracted_tissue
        tissue_column = "TISSUE"

    if cell_line_column and cell_line_column != "CELL_LINE_NAME":
        annotation_copy["CELL_LINE_NAME"] = annotation_copy[cell_line_column].astype(str)

    if tissue_column and tissue_column != "TISSUE":
        annotation_copy["TISSUE"] = annotation_copy[tissue_column].astype(str)

    return annotation_copy


def orient_expression_data(expression_df: pd.DataFrame, annotations_df: pd.DataFrame | None) -> pd.DataFrame:
    expression_copy = expression_df.copy()
    detected_cell_line = detect_column(expression_copy.columns.tolist(), COLUMN_CANDIDATES["cell_line"])
    detected_cosmic = detect_column(expression_copy.columns.tolist(), COLUMN_CANDIDATES["cosmic_id"])

    if detected_cell_line or detected_cosmic:
        if detected_cell_line and detected_cell_line != "CELL_LINE_NAME":
            expression_copy["CELL_LINE_NAME"] = expression_copy[detected_cell_line].astype(str)
        if detected_cosmic and detected_cosmic != "COSMIC_ID":
            expression_copy["COSMIC_ID"] = expression_copy[detected_cosmic].astype(str)
        return expression_copy

    # Some public transcriptomic resources store genes as rows and cell lines as columns.
    # We transpose into a cell-line-by-gene matrix because each row should represent one model system.
    transposed = expression_copy.copy()
    first_column = transposed.columns[0]
    transposed = transposed.set_index(first_column).T.reset_index().rename(columns={"index": "CELL_LINE_NAME"})

    if annotations_df is not None:
        annotations = parse_annotation_identifiers(annotations_df)
        annotation_columns = [col for col in ["CELL_LINE_NAME", "COSMIC_ID", "TISSUE"] if col in annotations.columns]
        if annotation_columns:
            transposed = transposed.merge(
                annotations[annotation_columns].drop_duplicates(subset=["CELL_LINE_NAME"]),
                on="CELL_LINE_NAME",
                how="left",
            )

    for column in transposed.columns:
        if column not in {"CELL_LINE_NAME", "COSMIC_ID", "TISSUE"}:
            transposed[column] = pd.to_numeric(transposed[column], errors="coerce")

    return transposed


def reshape_response_data(response_df: pd.DataFrame, annotations_df: pd.DataFrame | None) -> pd.DataFrame:
    response_copy = response_df.copy()
    columns = response_copy.columns.tolist()
    target_column = detect_column(columns, COLUMN_CANDIDATES["target"])
    drug_column = detect_column(columns, COLUMN_CANDIDATES["drug"])

    if target_column and drug_column:
        detected_cell_line = detect_column(columns, COLUMN_CANDIDATES["cell_line"])
        detected_cosmic = detect_column(columns, COLUMN_CANDIDATES["cosmic_id"])

        if detected_cell_line and detected_cell_line != "CELL_LINE_NAME":
            response_copy["CELL_LINE_NAME"] = response_copy[detected_cell_line].astype(str)
        if detected_cosmic and detected_cosmic != "COSMIC_ID":
            response_copy["COSMIC_ID"] = response_copy[detected_cosmic].astype(str)
        if drug_column != "DRUG_NAME":
            response_copy["DRUG_NAME"] = response_copy[drug_column].astype(str)

        cancer_column = detect_column(columns, COLUMN_CANDIDATES["cancer_type"])
        if cancer_column and cancer_column != "TISSUE":
            response_copy["TISSUE"] = response_copy[cancer_column].astype(str)

        return response_copy

    index_name = columns[0]
    wide_response = response_copy.set_index(index_name)
    long_response = (
        wide_response.reset_index()
        .melt(id_vars=index_name, var_name="DRUG_NAME", value_name="LN_IC50")
        .rename(columns={index_name: "COSMIC_ID"})
    )
    long_response["COSMIC_ID"] = long_response["COSMIC_ID"].astype(str)

    if annotations_df is not None:
        annotations = parse_annotation_identifiers(annotations_df)
        available_columns = [col for col in ["COSMIC_ID", "CELL_LINE_NAME", "TISSUE"] if col in annotations.columns]
        if available_columns:
            long_response = long_response.merge(
                annotations[available_columns].drop_duplicates(subset=["COSMIC_ID"]),
                on="COSMIC_ID",
                how="left",
            )

    return long_response


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, Any]:
    expression_path = LOCAL_EXPRESSION_PATH if LOCAL_EXPRESSION_PATH.exists() else download_to_cache(
        DEFAULT_DATA_URLS["expression"], CACHE_DIR / "gene_expression.csv.gz"
    )
    response_path = LOCAL_RESPONSE_PATH if LOCAL_RESPONSE_PATH.exists() else download_to_cache(
        DEFAULT_DATA_URLS["response"], CACHE_DIR / "drug_response.csv"
    )
    annotations_path = download_to_cache(DEFAULT_DATA_URLS["annotations"], CACHE_DIR / "cell_annotations.csv.gz")

    if not expression_path or not response_path:
        return {
            "ok": False,
            "message": "Unable to find local CSVs or download the default public datasets.",
            "expression_df": None,
            "response_df": None,
            "annotations_df": None,
        }

    try:
        raw_expression = read_table(expression_path)
        raw_response = read_table(response_path)
        annotations_df = read_table(annotations_path) if annotations_path else None
        expression_df = orient_expression_data(raw_expression, annotations_df)
        response_df = reshape_response_data(raw_response, annotations_df)
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Data loading failed: {exc}",
            "expression_df": None,
            "response_df": None,
            "annotations_df": None,
        }

    return {
        "ok": True,
        "message": "Data loaded successfully.",
        "expression_df": expression_df,
        "response_df": response_df,
        "annotations_df": annotations_df,
    }


def ensure_identifier_columns(expression_df: pd.DataFrame, response_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    expression = expression_df.copy()
    response = response_df.copy()

    expr_cell_line = detect_column(expression.columns.tolist(), COLUMN_CANDIDATES["cell_line"])
    expr_cosmic = detect_column(expression.columns.tolist(), COLUMN_CANDIDATES["cosmic_id"])
    resp_cell_line = detect_column(response.columns.tolist(), COLUMN_CANDIDATES["cell_line"])
    resp_cosmic = detect_column(response.columns.tolist(), COLUMN_CANDIDATES["cosmic_id"])

    if expr_cell_line and resp_cell_line:
        if expr_cell_line != "CELL_LINE_NAME":
            expression["CELL_LINE_NAME"] = expression[expr_cell_line].astype(str)
        if resp_cell_line != "CELL_LINE_NAME":
            response["CELL_LINE_NAME"] = response[resp_cell_line].astype(str)
        return expression, response, "CELL_LINE_NAME"

    if expr_cosmic and resp_cosmic:
        if expr_cosmic != "COSMIC_ID":
            expression["COSMIC_ID"] = expression[expr_cosmic].astype(str)
        if resp_cosmic != "COSMIC_ID":
            response["COSMIC_ID"] = response[resp_cosmic].astype(str)
        return expression, response, "COSMIC_ID"

    raise ValueError(
        "Could not detect a shared cell line identifier between the expression and drug response tables."
    )


def preprocess_data(
    expression_df: pd.DataFrame,
    response_df: pd.DataFrame,
    selected_drug: str,
    selected_cancer_type: str,
    top_n_genes: int,
) -> dict[str, Any]:
    expression, response, merge_key = ensure_identifier_columns(expression_df, response_df)

    drug_column = detect_column(response.columns.tolist(), COLUMN_CANDIDATES["drug"])
    cancer_column = detect_column(response.columns.tolist(), COLUMN_CANDIDATES["cancer_type"])
    target_column = detect_column(response.columns.tolist(), ["IC50"]) or detect_column(
        response.columns.tolist(), ["LN_IC50"]
    )

    if not drug_column:
        raise ValueError("Could not detect the drug name column in the drug response table.")
    if not cancer_column:
        raise ValueError("Could not detect the cancer or tissue type column in the drug response table.")
    if not target_column:
        raise ValueError("Could not detect an `IC50` or `LN_IC50` target column in the drug response table.")

    response = response[
        (response[drug_column].astype(str) == selected_drug)
        & (response[cancer_column].astype(str) == selected_cancer_type)
    ].copy()

    response[target_column] = pd.to_numeric(response[target_column], errors="coerce")
    response = response.dropna(subset=[target_column])

    # Merging gene expression with matched pharmacology measurements creates the modeling cohort:
    # each row corresponds to one cancer cell line with both molecular features and drug response.
    merged_df = expression.merge(response, on=merge_key, how="inner")
    if len(merged_df) < 12:
        raise ValueError("Too few matched cell lines after filtering. Try a broader drug or cancer selection.")

    expression_metadata_columns = {
        detect_column(expression.columns.tolist(), COLUMN_CANDIDATES["cell_line"]),
        detect_column(expression.columns.tolist(), COLUMN_CANDIDATES["cosmic_id"]),
        "CELL_LINE_NAME",
        "COSMIC_ID",
        "TISSUE",
    }
    expression_feature_candidates = [
        column for column in expression.columns if column not in expression_metadata_columns and column is not None
    ]
    expression_features = expression[expression_feature_candidates].apply(pd.to_numeric, errors="coerce")
    gene_columns = expression_features.select_dtypes(include=[np.number]).columns.tolist()
    if not gene_columns:
        raise ValueError("No numeric gene expression columns were found after merging the datasets.")

    # Only transcriptomic features are used here so that the model learns from baseline molecular state
    # rather than leaking response-side metadata into the prediction task.
    x = merged_df[gene_columns].copy()
    y = merged_df[target_column].copy()

    # Low-variance genes contribute little signal across the cohort and can add noise.
    variance_filter = VarianceThreshold(threshold=0.01)
    x_var = variance_filter.fit_transform(x)
    kept_after_variance = x.columns[variance_filter.get_support()].tolist()
    if len(kept_after_variance) < 5:
        raise ValueError("Low-variance filtering removed too many genes to build a stable model.")

    # Univariate feature selection highlights genes whose expression is most associated with IC50 variation.
    effective_k = min(top_n_genes, len(kept_after_variance), max(5, len(merged_df) - 2))
    selector = SelectKBest(score_func=f_regression, k=effective_k)
    x_selected = selector.fit_transform(x_var, y)
    selected_genes = pd.Index(kept_after_variance)[selector.get_support()].tolist()

    selected_df = pd.DataFrame(x_selected, columns=selected_genes, index=merged_df.index)
    if len(selected_df) < 10 or selected_df.shape[1] == 0:
        raise ValueError("Selected feature set is empty or sample count is too small for train/test splitting.")

    test_size = 0.2
    if len(selected_df) * test_size < 2:
        raise ValueError("Invalid train/test split for the current filtered cohort. Please choose a larger cohort.")

    x_train, x_test, y_train, y_test = train_test_split(
        selected_df,
        y,
        test_size=test_size,
        random_state=42,
    )
    if len(x_train) < 5 or len(x_test) < 2:
        raise ValueError("Train/test split produced too few samples for reliable evaluation.")

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "selected_genes": selected_genes,
        "target_column": target_column,
        "filtered_df": merged_df,
        "selected_feature_df": selected_df,
    }


def train_model(processed: dict[str, Any]) -> dict[str, Any]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(processed["x_train"])
    x_test_scaled = scaler.transform(processed["x_test"])

    # Standardization keeps gene expression features on comparable scales before learning.
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train_scaled, processed["y_train"])
    predictions = model.predict(x_test_scaled)

    mse = mean_squared_error(processed["y_test"], predictions)
    r2 = r2_score(processed["y_test"], predictions)

    importance_df = pd.DataFrame(
        {
            "Gene": processed["selected_genes"],
            "Feature Importance": model.feature_importances_,
        }
    ).sort_values("Feature Importance", ascending=False)

    result = {
        "model": model,
        "scaler": scaler,
        "selected_genes": processed["selected_genes"],
        "target_column": processed["target_column"],
        "mse": mse,
        "r2": r2,
        "x_test": processed["x_test"],
        "y_test": processed["y_test"],
        "predictions": predictions,
        "feature_importance_df": importance_df,
        "filtered_df": processed["filtered_df"],
        "selected_feature_df": processed["selected_feature_df"],
    }

    st.session_state["trained_model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["selected_genes"] = processed["selected_genes"]
    st.session_state["feature_importance_df"] = importance_df
    st.session_state["target_column"] = processed["target_column"]
    st.session_state["test_predictions"] = predictions
    st.session_state["test_actuals"] = processed["y_test"]
    st.session_state["selected_feature_df"] = processed["selected_feature_df"]
    st.session_state["filtered_df"] = processed["filtered_df"]

    return result


def make_prediction_plot(actual: pd.Series, predicted: np.ndarray, r2_value: float, target_label: str) -> go.Figure:
    plot_df = pd.DataFrame({"Actual": actual, "Predicted": predicted})
    figure = px.scatter(
        plot_df,
        x="Actual",
        y="Predicted",
        color="Predicted",
        color_continuous_scale="Tealgrn",
        opacity=0.82,
    )
    if len(plot_df) >= 2:
        slope, intercept = np.polyfit(plot_df["Actual"], plot_df["Predicted"], 1)
        trend_x = np.linspace(plot_df["Actual"].min(), plot_df["Actual"].max(), 100)
        trend_y = slope * trend_x + intercept
        figure.add_trace(
            go.Scatter(
                x=trend_x,
                y=trend_y,
                mode="lines",
                name="Trendline",
                line=dict(color="#f97316", width=3),
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[plot_df["Actual"].min(), plot_df["Actual"].max()],
            y=[plot_df["Actual"].min(), plot_df["Actual"].max()],
            mode="lines",
            name="Ideal fit",
            line=dict(color="#93c5fd", dash="dash"),
        )
    )
    figure.update_layout(
        title=f"Predicted vs Actual {target_label} (R² = {r2_value:.3f})",
        template="plotly_dark",
        xaxis_title=f"Actual {target_label}",
        yaxis_title=f"Predicted {target_label}",
        height=580,
        margin=dict(l=56, r=36, t=84, b=58),
        title_x=0.03,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(automargin=True, title_standoff=16)
    figure.update_yaxes(automargin=True, title_standoff=16)
    return figure


def make_distribution_plot(actual: pd.Series, predicted: np.ndarray, target_label: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=actual,
            name=f"Actual {target_label}",
            marker_color="#14b8a6",
            opacity=0.65,
            nbinsx=20,
        )
    )
    figure.add_trace(
        go.Histogram(
            x=predicted,
            name=f"Predicted {target_label}",
            marker_color="#60a5fa",
            opacity=0.65,
            nbinsx=20,
        )
    )
    figure.update_layout(
        barmode="overlay",
        template="plotly_dark",
        title=f"{target_label} Distribution Comparison",
        xaxis_title=target_label,
        yaxis_title="Count",
        height=520,
        margin=dict(l=56, r=36, t=84, b=58),
        title_x=0.03,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.08,
    )
    figure.update_xaxes(automargin=True, title_standoff=16)
    figure.update_yaxes(automargin=True, title_standoff=16)
    return figure


def make_feature_importance_plot(feature_df: pd.DataFrame) -> go.Figure:
    top_features = feature_df.head(10).sort_values("Feature Importance", ascending=True)
    figure = px.bar(
        top_features,
        x="Feature Importance",
        y="Gene",
        orientation="h",
        color="Feature Importance",
        color_continuous_scale="Viridis",
    )
    figure.update_layout(
        template="plotly_dark",
        title="Top 10 Genomic Biomarkers",
        xaxis_title="Random Forest Importance",
        yaxis_title="Gene",
        height=520,
        margin=dict(l=56, r=36, t=84, b=58),
        title_x=0.03,
        font=dict(size=14),
        coloraxis_showscale=False,
    )
    figure.update_xaxes(automargin=True, title_standoff=16)
    figure.update_yaxes(automargin=True, title_standoff=16)
    return figure


@st.cache_data(show_spinner=False)
def fetch_gene_summary(gene_name: str) -> str:
    query_url = "https://mygene.info/v3/query"
    params = {
        "q": gene_name,
        "species": "human",
        "fields": "symbol,name,summary",
        "size": 1,
    }
    headers = {"User-Agent": "OncoResponse-AI/1.0"}
    try:
        response = requests.get(query_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
        hits = payload.get("hits", [])
        if not hits:
            return "Function not available."
        summary = hits[0].get("summary") or hits[0].get("name")
        if not summary:
            return "Function not available."
        return str(summary)
    except (requests.RequestException, ValueError, KeyError):
        return "Function not available."


def render_metric_cards(result: dict[str, Any]) -> None:
    metric_columns = st.columns(4)
    metrics = [
        ("Mean Squared Error", f"{result['mse']:.4f}"),
        ("R² Score", f"{result['r2']:.4f}"),
        ("Cell Lines Used", f"{len(result['filtered_df'])}"),
        ("Genes Used", f"{len(result['selected_genes'])}"),
    ]
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown("</div>", unsafe_allow_html=True)


def render_biomarker_table(feature_df: pd.DataFrame) -> None:
    biomarker_df = feature_df.head(10).copy()
    biomarker_df["Biological Function"] = biomarker_df["Gene"].apply(fetch_gene_summary)
    st.dataframe(
        biomarker_df[["Gene", "Feature Importance", "Biological Function"]],
        use_container_width=True,
        hide_index=True,
    )


def render_what_if_simulator(result: dict[str, Any]) -> None:
    feature_df = result["feature_importance_df"]
    top_genes = feature_df.head(3)["Gene"].tolist()
    if len(top_genes) < 3:
        st.info("At least three informative genes are needed to render the sensitivity simulator.")
        return

    st.markdown('<div class="section-label">What-If Sensitivity Simulator</div>', unsafe_allow_html=True)
    st.caption("Adjust the top three biomarker expression values and simulate the expected drug response.")

    feature_frame = result["selected_feature_df"][result["selected_genes"]]
    # Median expression values create a biologically reasonable baseline synthetic cell line.
    baseline = feature_frame.median().to_dict()

    synthetic_profile = {gene: float(baseline[gene]) for gene in result["selected_genes"]}
    slider_columns = st.columns(3)
    for index, gene in enumerate(top_genes):
        gene_values = feature_frame[gene].dropna()
        default_value = float(gene_values.median())
        min_value = float(gene_values.min())
        max_value = float(gene_values.max())
        if min_value == max_value:
            min_value -= 1.0
            max_value += 1.0
        with slider_columns[index]:
            synthetic_profile[gene] = st.slider(
                f"{gene} expression",
                min_value=min_value,
                max_value=max_value,
                value=default_value,
            )

    simulator_input = pd.DataFrame([synthetic_profile], columns=result["selected_genes"])
    scaled_input = result["scaler"].transform(simulator_input)
    synthetic_prediction = float(result["model"].predict(scaled_input)[0])

    prediction_col, explanation_col = st.columns([1, 2])
    with prediction_col:
        st.metric(f"Predicted {result['target_column']}", f"{synthetic_prediction:.4f}")
    with explanation_col:
        st.info("Higher predicted IC50 generally suggests lower drug sensitivity or greater resistance.")


def render_sidebar(expression_df: pd.DataFrame, response_df: pd.DataFrame) -> tuple[str, str, int, bool]:
    st.sidebar.markdown("## Model Controls")
    st.sidebar.caption("Tune the cohort and biomarker panel before training.")

    _, response_with_ids, _ = ensure_identifier_columns(expression_df, response_df)
    drug_column = detect_column(response_with_ids.columns.tolist(), COLUMN_CANDIDATES["drug"])
    cancer_column = detect_column(response_with_ids.columns.tolist(), COLUMN_CANDIDATES["cancer_type"])

    if not drug_column or not cancer_column:
        raise ValueError("The response table must include detectable drug and cancer type columns.")

    available_drugs = sorted(response_with_ids[drug_column].dropna().astype(str).unique().tolist())
    available_cancers = sorted(response_with_ids[cancer_column].dropna().astype(str).unique().tolist())

    if not available_drugs or not available_cancers:
        raise ValueError("No selectable drugs or cancer types were found in the loaded dataset.")

    selected_drug = st.sidebar.selectbox("Select drug", available_drugs)
    selected_cancer_type = st.sidebar.selectbox("Select cancer type", available_cancers)
    top_n_genes = st.sidebar.slider("Number of top genes/features", min_value=10, max_value=500, value=100, step=10)
    train_clicked = st.sidebar.button("Train Model", type="primary", use_container_width=True)
    return selected_drug, selected_cancer_type, top_n_genes, train_clicked


def initialize_session_state() -> None:
    for key in [
        "trained_model",
        "scaler",
        "selected_genes",
        "feature_importance_df",
        "target_column",
        "test_predictions",
        "test_actuals",
        "selected_feature_df",
        "filtered_df",
        "training_result",
    ]:
        st.session_state.setdefault(key, None)


def main() -> None:
    inject_styles()
    initialize_session_state()

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.35rem;">OncoResponse AI</h1>
            <p style="margin:0;font-size:1.08rem;color:#cbd5e1;">
                Predicting cancer drug response from genomic signatures
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_bundle = load_data()
    if not data_bundle["ok"]:
        st.warning(data_bundle["message"])
        with st.expander("Expected file structure"):
            st.markdown(EXPECTED_STRUCTURE_MESSAGE)
        return

    expression_df = data_bundle["expression_df"]
    response_df = data_bundle["response_df"]
    if expression_df is None or response_df is None:
        st.warning("The datasets could not be prepared for modeling.")
        with st.expander("Expected file structure"):
            st.markdown(EXPECTED_STRUCTURE_MESSAGE)
        return

    st.markdown('<div class="section-label">Dashboard Inputs</div>', unsafe_allow_html=True)
    st.caption("Select a therapy context, train the model, then inspect the strongest genomic biomarkers.")

    try:
        selected_drug, selected_cancer_type, top_n_genes, train_clicked = render_sidebar(expression_df, response_df)
    except ValueError as exc:
        st.error(str(exc))
        with st.expander("Expected file structure"):
            st.markdown(EXPECTED_STRUCTURE_MESSAGE)
        return

    if train_clicked:
        try:
            processed = preprocess_data(
                expression_df=expression_df,
                response_df=response_df,
                selected_drug=selected_drug,
                selected_cancer_type=selected_cancer_type,
                top_n_genes=top_n_genes,
            )
            st.session_state["training_result"] = train_model(processed)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected training failure: {exc}")

    result = st.session_state.get("training_result")
    if not result:
        st.info("Choose a drug, cancer type, and feature count in the sidebar, then click `Train Model`.")
        with st.expander("Data source notes"):
            st.markdown(
                """
                - The app uses local `gene_expression.csv` and `drug_response.csv` if they exist in the project folder.
                - Otherwise, it tries the built-in online GDSC-compatible sources and caches them in `data_cache/`.
                - If the online sources are unavailable, you can add the CSVs manually using the expected schema.
                """
            )
        return

    render_metric_cards(result)

    prediction_tab, distribution_tab = st.tabs(["Prediction Fit", "Response Distribution"])
    with prediction_tab:
        st.plotly_chart(
            make_prediction_plot(result["y_test"], result["predictions"], result["r2"], result["target_column"]),
            use_container_width=True,
        )
    with distribution_tab:
        st.plotly_chart(
            make_distribution_plot(result["y_test"], result["predictions"], result["target_column"]),
            use_container_width=True,
        )

    st.plotly_chart(make_feature_importance_plot(result["feature_importance_df"]), use_container_width=True)
    st.markdown('<div class="section-label">Biomarker Discovery Module</div>', unsafe_allow_html=True)
    st.caption("Top candidate biomarkers ranked by model importance, with lightweight gene annotations.")
    render_biomarker_table(result["feature_importance_df"])

    render_what_if_simulator(result)


if __name__ == "__main__":
    main()
