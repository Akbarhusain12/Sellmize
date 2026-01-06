import os
import logging
import pandas as pd
from flask import current_app

from app.ml.sku_health import analyze_sku_health  # adjust path if different

logger = logging.getLogger(__name__)


def load_analysis_from_file(filename):
    """
    Load analysis data from generated Excel report.
    Rebuilds derived analytics that were not stored directly.
    """
    try:
        output_path = os.path.join(current_app.config['OUTPUT_FOLDER'], filename)

        if not os.path.exists(output_path):
            logger.error(f"File not found: {output_path}")
            return None

        logger.info(f"Loading analysis from: {output_path}")

        # ---------------- SUMMARY ----------------
        summary_df = pd.read_excel(output_path, sheet_name="Summary")

        # ---------------- TOP SKUS ----------------
        try:
            top_10_skus = pd.read_excel(output_path, sheet_name="Top 10 SKUs")
            top_10_skus = top_10_skus.where(pd.notnull(top_10_skus), None).to_dict("records")
        except Exception:
            top_10_skus = []

        # ---------------- TOP RETURNS ----------------
        try:
            top_10_returns_df = pd.read_excel(output_path, sheet_name="Top 10 Returns")
            top_10_returns = top_10_returns_df.where(pd.notnull(top_10_returns_df), None).to_dict("records")
        except Exception:
            top_10_returns_df = pd.DataFrame()
            top_10_returns = []

        # ---------------- TOP STATES ----------------
        try:
            state_df = pd.read_excel(output_path, sheet_name="Top 10 States")
            if "ship_state" in state_df.columns:
                state_df = state_df.rename(
                    columns={"ship_state": "state", "quantity": "total_orders"}
                )
            top_states = state_df.where(pd.notnull(state_df), None).to_dict("records")
        except Exception:
            top_states = []

        # ---------------- UNPAID ORDERS ----------------
        try:
            unpaid_orders_df = pd.read_excel(output_path, sheet_name="Unpaid Orders")
            unpaid_orders = unpaid_orders_df.where(pd.notnull(unpaid_orders_df), None).to_dict("records")
        except Exception:
            unpaid_orders_df = pd.DataFrame()
            unpaid_orders = []

        # ---------------- MERGED DATA + SKU HEALTH ----------------
        sku_health_rows = []
        try:
            merged_df = pd.read_excel(output_path, sheet_name="Merged Data")

            health = analyze_sku_health(
                merged_data=merged_df,
                returns=top_10_returns_df,
                unpaid_orders=unpaid_orders_df,
                min_orders=1
            )

            scored = health.scored_df
            scored["health_score_pred"] = scored["health_score_pred"].round(2)
            scored["proxy_label"] = scored["proxy_label"].round(2)

            def rate(x):
                if x >= 75: return "Excellent"
                if x >= 60: return "Good"
                if x >= 40: return "Fair"
                return "Poor"

            scored["health_rating"] = scored["health_score_pred"].apply(rate)

            scored = scored.rename(columns={
                "SKU": "SKU",
                "health_score_pred": "Score",
                "proxy_label": "Rating",
                "health_rating": "Health",
                "key_issues": "Key_Issues",
                "cluster": "Segment",
                "anomaly": "Anomaly",
            })

            if "Anomaly" in scored.columns:
                scored["Anomaly"] = scored["Anomaly"].map({True: "Yes", False: "No"})

            sku_health_df = scored[["SKU", "Score", "Health", "Key_Issues", "Rating", "Anomaly"]]
            sku_health_rows = sku_health_df.where(pd.notnull(sku_health_df), None).to_dict("records")

        except Exception as e:
            logger.warning(f"SKU health recomputation failed: {e}")

        # ---------------- SUMMARY DICT ----------------
        summary_dict = {}
        for _, row in summary_df.iterrows():
            metric = str(row["Metric"]).strip()
            value = row["Value"]

            try:
                value = float(value)
                if pd.isna(value):  # Check if it turned into NaN
                    value = 0.0
            except Exception:
                value = 0.0

            mapping = {
                "Total Quantity": "total_quantity",
                "Total Return Quantity": "total_return_quantity",
                "Total Unpaid Orders": "total_unpaid_orders",
                "Total Payment": "total_payment",
                "Total Cost": "total_cost",
                "Total Amz Fees": "total_amz_fees",
                "Net Profit": "net_profit",
            }

            if metric in mapping:
                summary_dict[mapping[metric]] = value
            else:
                summary_dict["expense_" + metric.lower().replace(" ", "_")] = value

        return {
            "summary_data": summary_dict,
            "top_10_skus": top_10_skus,
            "top_10_returns": top_10_returns,
            "top_states": top_states,
            "unpaid_orders": unpaid_orders,
            "sku_health": sku_health_rows,
        }

    except Exception as e:
        logger.exception(f"Failed loading analysis from file: {e}")
        return None
