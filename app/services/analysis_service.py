import os
import logging
import pandas as pd
from flask import current_app

logger = logging.getLogger(__name__)


def load_analysis_from_file(filename):
    try:
        output_path = os.path.join(current_app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(output_path):
            return None

        # --- 1. Load Summary (Mandatory) ---
        summary_df = pd.read_excel(output_path, sheet_name="Summary")
        
        # --- 2. Load Top SKUs with Fallback ---
        try:
            top_10_skus_df = pd.read_excel(output_path, sheet_name="Top 10 SKUs")
        except Exception:
            # SAFETY VALVE: Re-calculate from Merged Data if sheet is missing
            logger.warning("Top 10 SKUs sheet missing. Re-calculating from Merged Data.")
            main_df = pd.read_excel(output_path, sheet_name="Merged Data")
            top_10_skus_df = main_df.groupby('sku')['quantity'].sum().nlargest(10).reset_index()
            top_10_skus_df.columns = ['sku', 'quantity']
        
        top_10_skus = top_10_skus_df.where(pd.notnull(top_10_skus_df), None).to_dict("records")

        # --- 3. Load Top Returns ---
        # NOTE: We now check for 'raw_return_reason' which we added to ecommerce_service
        try:
            top_10_returns_df = pd.read_excel(output_path, sheet_name="Top 10 Returns")
        except Exception:
            logger.warning("Top 10 Returns sheet missing. Checking Merged Data.")
            main_df = pd.read_excel(output_path, sheet_name="Merged Data")
            # If we have the new raw_return_reason column, use it!
            if 'raw_return_reason' in main_df.columns:
                top_10_returns_df = main_df['raw_return_reason'].value_counts().nlargest(10).reset_index()
                top_10_returns_df.columns = ['reason', 'count']
            else:
                top_10_returns_df = pd.DataFrame()
        
        top_10_returns = top_10_returns_df.where(pd.notnull(top_10_returns_df), None).to_dict("records")

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

        # ---------------- SUMMARY DICT ----------------
        summary_dict = {}
        for _, row in summary_df.iterrows():
            metric = str(row["Metric"]).strip()
            # Handle potential NaN in Value column
            value = 0.0 if pd.isna(row["Value"]) else row["Value"]
            
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
                # Store dynamic expenses with a clean key
                clean_metric = metric.lower().replace(" ", "_").replace("-", "_")
                summary_dict[f"expense_{clean_metric}"] = value

        return {
            "summary_data": summary_dict,
            "top_10_skus": top_10_skus,
            "top_10_returns": top_10_returns,
            "top_states": top_states,
            "unpaid_orders": unpaid_orders
        }

    except Exception as e:
        logger.exception(f"Critical failure loading analysis file: {e}")
        return None
