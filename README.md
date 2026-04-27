# OncoResponse AI

OncoResponse AI is a Streamlit dashboard for exploring cancer drug sensitivity from gene expression data. It links genomics and pharmacology by training a machine learning model to predict drug response and then surfaces the most informative biomarkers behind that prediction.

## What the app does

- Downloads default GDSC-compatible public datasets automatically, with local CSVs supported as overrides
- Lets you select a drug, a cancer or tissue type, and the number of top genes to use
- Trains a `RandomForestRegressor` to predict `IC50` or `LN_IC50`
- Reports model quality with Mean Squared Error and R²
- Visualizes predicted vs actual response, response distributions, and top gene importances
- Annotates leading biomarker genes with short biological summaries from MyGene.info
- Includes a what-if simulator that estimates how changing the top three gene expression values affects predicted drug response

## Expected data format

The app first checks for these local files in the project directory:

### `gene_expression.csv`

- Rows should represent cell lines
- Columns should represent genes
- Include at least one identifier column such as `CELL_LINE_NAME` or `COSMIC_ID`

Example:

| CELL_LINE_NAME | COSMIC_ID | EGFR | BRAF | ERBB2 |
| --- | --- | --- | --- | --- |
| CAL-120 | 906826 | 5.21 | 3.84 | 4.10 |
| DMS-114 | 687983 | 2.91 | 5.22 | 3.48 |

### `drug_response.csv`

- Rows should represent cell line and drug pairs
- Include a cell line identifier matching the expression table
- Include a drug name column
- Include a cancer or tissue type column
- Include an `IC50` or `LN_IC50` response column

Example:

| CELL_LINE_NAME | COSMIC_ID | DRUG_NAME | TISSUE | LN_IC50 |
| --- | --- | --- | --- | --- |
| CAL-120 | 906826 | Erlotinib | Breast | 1.72 |
| DMS-114 | 687983 | Erlotinib | Lung | 3.45 |

If local files are absent, the app tries to download built-in public GDSC-compatible sources and caches them under `data_cache/`.

## How to run

1. Create and activate a Python 3.10+ environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. Open the local Streamlit URL shown in the terminal

## IC50 explained

IC50 is the concentration of a drug required to inhibit a biological process or cell viability by 50%. In pharmacology, a lower IC50 often indicates that a drug is more potent or that a cell line is more sensitive to that therapy. A higher predicted IC50 generally suggests lower drug sensitivity or greater resistance.

## Feature importance explained

The model uses Random Forest feature importances to rank genes by how much they contribute to predictive performance. These scores help highlight candidate biomarkers associated with drug response, but they do not prove mechanism. They are best treated as hypothesis-generating signals for follow-up analysis.

## Modeling workflow

1. Load gene expression and drug response data
2. Detect likely identifier, drug, tissue, and target columns
3. Merge the datasets by shared cell line identifier
4. Filter to the selected drug and cancer type
5. Keep numeric gene expression features only
6. Remove low-variance genes
7. Select the top `N` genes with `SelectKBest(f_regression)`
8. Standardize features and train a `RandomForestRegressor`
9. Evaluate predictions on a held-out test split
10. Interpret the top biomarkers and simulate gene-expression changes

## Limitations

- Feature importance reflects predictive association, not causation
- Biomarker rankings should not be treated as clinically validated findings
- Public dataset preprocessing choices can influence model performance
- The built-in downloader depends on external URLs being available
- Gene annotation lookups may fail or return incomplete summaries
- Results require experimental and biological validation before any translational use
