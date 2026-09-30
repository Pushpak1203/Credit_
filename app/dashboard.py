import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.credit_sense.decision import score_row


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Credit Sense Prototype",
    page_icon="💳",
    layout="wide",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>
        .credit-sense-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0 20px 0;
            font-size: 14px;
        }

        .credit-sense-table th,
        .credit-sense-table td {
            border: 1px solid #d9d9d9;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
        }

        .credit-sense-table th {
            font-weight: 600;
            background-color: rgba(128, 128, 128, 0.12);
        }

        .credit-sense-table tr:nth-child(even) {
            background-color: rgba(128, 128, 128, 0.05);
        }

        .prototype-banner {
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #e0a800;
            background-color: rgba(255, 193, 7, 0.10);
            margin-bottom: 20px;
        }

        .metric-label {
            font-size: 14px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# PROTOTYPE DISCLAIMER
# =============================================================================

st.warning(
    "PROTOTYPE / SYNTHETIC DATA ONLY — "
    "This system has not been validated for real lending and must not be "
    "used to make real credit decisions."
)

st.title("Credit Sense — Offline Credit Decisioning Prototype")

st.caption(
    "Alternative-data credit decisioning prototype for U.S. community "
    "lending scenarios. All data is synthetic."
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json(path: Path):
    """Load JSON from disk."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path: Path):
    """Load CSV from disk."""
    if not path.exists():
        return None

    return pd.read_csv(path)


def load_markdown(path: Path):
    """Load Markdown text from disk."""
    if not path.exists():
        return None

    return path.read_text(encoding="utf-8")


def render_dataframe_as_html(df: pd.DataFrame):
    """
    Render a pandas DataFrame as HTML.

    IMPORTANT:
    Do not use st.dataframe() or st.table() here.

    Both Streamlit components can invoke PyArrow in modern Streamlit
    versions. This environment has a Windows Application Control
    restriction that blocks PyArrow's native DLL.

    Pandas HTML rendering avoids PyArrow completely.
    """

    if df is None or df.empty:
        st.info("No records available.")
        return

    html = df.to_html(
        index=False,
        escape=True,
        classes="credit-sense-table",
        border=0,
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def show_csv_table(path: Path, empty_message: str):
    """Load a CSV and render it without Streamlit's Arrow components."""

    df = load_csv(path)

    if df is None:
        st.info(empty_message)
        return

    render_dataframe_as_html(df)


# =============================================================================
# TABS
# =============================================================================

tabs = st.tabs(
    [
        "Applicant Scoring",
        "Model Performance",
        "Fairness",
        "Pilot Stage Comparison",
        "Shadow Mode",
        "Appeals Queue",
    ]
)


# =============================================================================
# TAB 1 — APPLICANT SCORING
# =============================================================================

with tabs[0]:

    st.subheader("Applicant Scoring")

    st.write(
        "Enter synthetic applicant information and run the Credit Sense "
        "decision engine."
    )

    # -------------------------------------------------------------------------
    # INPUT COLUMNS
    # -------------------------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    # =========================================================================
    # COLUMN 1 — CREDIT / REPAYMENT
    # =========================================================================

    with c1:

        st.markdown("### Credit & Repayment")

        credit_score = st.number_input(
            "Credit score",
            min_value=500,
            max_value=820,
            value=650,
            step=1,
            help="Synthetic bureau-style credit score.",
        )

        credit_history_years = st.number_input(
            "Credit history years",
            min_value=0.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
        )

        on_time_ratio = st.slider(
            "On-time payment ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.92,
            step=0.01,
        )

        delinq = st.number_input(
            "Delinquencies in last 24 months",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )

        months = st.number_input(
            "Months since last delinquency",
            min_value=0.0,
            max_value=60.0,
            value=36.0,
            step=1.0,
        )

    # =========================================================================
    # COLUMN 2 — ALTERNATIVE DATA
    # =========================================================================

    with c2:

        st.markdown("### Alternative Data")

        utility = st.slider(
            "Utility/rent payment regularity",
            min_value=0.0,
            max_value=1.0,
            value=0.92,
            step=0.01,
        )

        incvol = st.slider(
            "Income volatility",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
        )

        inflow = st.number_input(
            "Average monthly inflow",
            min_value=100.0,
            max_value=50000.0,
            value=5000.0,
            step=100.0,
        )

        overdraft = st.number_input(
            "Overdraft count",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
        )

        savings = st.number_input(
            "Savings buffer months",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5,
        )

    # =========================================================================
    # COLUMN 3 — EMPLOYMENT / LOAN
    # =========================================================================

    with c3:

        st.markdown("### Employment & Loan")

        gig = st.slider(
            "Gig income share",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
        )

        tenure = st.number_input(
            "Employment tenure years",
            min_value=0.0,
            max_value=30.0,
            value=4.0,
            step=0.5,
        )

        dti = st.slider(
            "Debt-to-income ratio (DTI)",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
        )

        amount = st.number_input(
            "Requested amount",
            min_value=100.0,
            max_value=50000.0,
            value=5000.0,
            step=100.0,
        )

        term = st.selectbox(
            "Term months",
            [6, 9, 12, 18, 24, 36],
        )

    # =========================================================================
    # SCORE BUTTON
    # =========================================================================

    st.divider()

    if st.button(
        "Score Applicant",
        type="primary",
        width="stretch",
    ):

        row = {
            "credit_score": credit_score,
            "credit_history_years": credit_history_years,
            "on_time_ratio": on_time_ratio,
            "delinquencies_24m": delinq,
            "months_since_last_delinquency": months,
            "utility_rent_regularity": utility,
            "income_volatility": incvol,
            "avg_monthly_inflow": inflow,
            "overdraft_count": overdraft,
            "savings_buffer_months": savings,
            "gig_income_share": gig,
            "employment_tenure_years": tenure,
            "dti": dti,
            "requested_amount": amount,
            "term_months": term,

            # Prototype defaults.
            "thin_file": 0,
            "no_file": 0,
        }

        try:

            result = score_row(row)

            # -----------------------------------------------------------------
            # DECISION METRICS
            # -----------------------------------------------------------------

            st.subheader("Decision Result")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Probability of Default",
                    f'{result["pd"]:.1%}',
                )

            with m2:
                st.metric(
                    "Decision",
                    result["decision"],
                )

            with m3:
                st.metric(
                    "Confidence",
                    f'{result["confidence"]:.1%}',
                )

            # -----------------------------------------------------------------
            # MODEL VERSION
            # -----------------------------------------------------------------

            if "model_version" in result:

                st.caption(
                    f'Model version: {result["model_version"]}'
                )

            # -----------------------------------------------------------------
            # REASON CODES
            # -----------------------------------------------------------------

            st.subheader(
                "Adverse-Action / Risk Reason Codes"
            )

            st.json(
                result["reason_codes"]
            )

            # -----------------------------------------------------------------
            # BORROWER MESSAGE
            # -----------------------------------------------------------------

            st.subheader(
                "Borrower-Facing Explanation"
            )

            st.info(
                result["borrower_message"]
            )

        except Exception as exc:

            st.error(
                "The scoring engine could not process this applicant."
            )

            st.exception(exc)


# =============================================================================
# TAB 2 — MODEL PERFORMANCE
# =============================================================================

with tabs[1]:

    st.subheader("Model Performance")

    st.caption(
        "Performance metrics are based entirely on synthetic test data."
    )

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    metrics_path = (
        ROOT
        / "outputs"
        / "metrics.json"
    )

    metrics = load_json(metrics_path)

    if metrics is not None:

        st.json(metrics)

    else:

        st.warning(
            "metrics.json was not found. "
            "Run the training and evaluation pipeline first."
        )

    # -------------------------------------------------------------------------
    # ROC CURVES
    # -------------------------------------------------------------------------

    roc_path = (
        ROOT
        / "outputs"
        / "plots"
        / "roc_curves.png"
    )

    if roc_path.exists():

        st.subheader("ROC Curves")

        st.image(
            str(roc_path),
            width="stretch",
        )

    else:

        st.info(
            "ROC curve has not been generated yet."
        )

    # -------------------------------------------------------------------------
    # CALIBRATION
    # -------------------------------------------------------------------------

    calibration_path = (
        ROOT
        / "outputs"
        / "plots"
        / "calibration.png"
    )

    if calibration_path.exists():

        st.subheader("Calibration Curve")

        st.image(
            str(calibration_path),
            width="stretch",
        )

    else:

        st.info(
            "Calibration plot has not been generated yet."
        )


# =============================================================================
# TAB 3 — FAIRNESS
# =============================================================================

with tabs[2]:

    st.subheader("Fairness Analysis")

    st.caption(
        "Protected attributes are used exclusively for fairness auditing "
        "and are excluded from model features."
    )

    fairness_path = (
        ROOT
        / "outputs"
        / "fairness_report.md"
    )

    fairness_report = load_markdown(
        fairness_path
    )

    if fairness_report is not None:

        st.markdown(
            fairness_report
        )

    else:

        st.warning(
            "Fairness report not found. "
            "Run the fairness pipeline first."
        )


# =============================================================================
# TAB 4 — PILOT STAGE COMPARISON
# =============================================================================

with tabs[3]:

    st.subheader(
        "Pilot Stage Comparison"
    )

    st.caption(
        "Comparison of incumbent underwriting, deterministic rules, "
        "traditional ML, alternative-data ML, and the Credit Sense "
        "hybrid workflow."
    )

    pilot_path = (
        ROOT
        / "outputs"
        / "pilot_comparison.csv"
    )

    show_csv_table(
        pilot_path,
        (
            "Pilot comparison output not found. "
            "Run: python -m src.credit_sense.pilot_simulation"
        ),
    )


# =============================================================================
# TAB 5 — SHADOW MODE
# =============================================================================

with tabs[4]:

    st.subheader(
        "Shadow Mode"
    )

    st.caption(
        "Shadow mode compares Credit Sense decisions with incumbent "
        "decisions without acting on the Credit Sense decision."
    )

    shadow_path = (
        ROOT
        / "outputs"
        / "shadow_disagreement.csv"
    )

    show_csv_table(
        shadow_path,
        (
            "Shadow-mode output not found. "
            "Run: python -m src.credit_sense.shadow"
        ),
    )


# =============================================================================
# TAB 6 — APPEALS QUEUE
# =============================================================================

with tabs[5]:

    st.subheader(
        "Appeals Queue"
    )

    st.caption(
        "Prototype appeal workflow backed by the local SQLite database."
    )

    try:

        from src.credit_sense.appeals import list_appeals

        appeals = list_appeals()

        appeals_df = pd.DataFrame(
            appeals,
            columns=[
                "id",
                "applicant_ref",
                "reason",
                "status",
                "reviewer_note",
                "created_at",
                "updated_at",
            ],
        )

        if appeals_df.empty:

            st.info(
                "No appeals are currently in the queue."
            )

        else:

            # IMPORTANT:
            # Do NOT use st.dataframe() or st.table().
            # Both can trigger PyArrow.
            render_dataframe_as_html(
                appeals_df
            )

    except Exception as exc:

        st.error(
            "The appeals queue could not be loaded."
        )

        st.exception(exc)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Credit Sense Prototype | Synthetic data only | "
    "Not validated for real lending | Offline execution"
)