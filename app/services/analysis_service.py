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
            logger.warning("Top 10 SKUs sheet missing. Re-calculating from Merged Data.")
            main_df = pd.read_excel(output_path, sheet_name="Merged Data")
            top_10_skus_df = main_df.groupby('sku')['quantity'].sum().nlargest(10).reset_index()
            top_10_skus_df.columns = ['sku', 'quantity']
        
        top_10_skus = top_10_skus_df.where(pd.notnull(top_10_skus_df), None).to_dict("records")

        # --- 3. Load Top Returns ---
        try:
            top_10_returns_df = pd.read_excel(output_path, sheet_name="Top 10 Returns")
        except Exception:
            logger.warning("Top 10 Returns sheet missing. Checking Merged Data.")
            main_df = pd.read_excel(output_path, sheet_name="Merged Data")
            if 'raw_return_reason' in main_df.columns:
                top_10_returns_df = main_df['raw_return_reason'].value_counts().nlargest(10).reset_index()
                top_10_returns_df.columns = ['reason', 'count']
            else:
                top_10_returns_df = pd.DataFrame()
        
        top_10_returns = top_10_returns_df.where(pd.notnull(top_10_returns_df), None).to_dict("records")

        # --- 4. Load Top States ---
        try:
            state_df = pd.read_excel(output_path, sheet_name="Top 10 States")
            if "ship_state" in state_df.columns:
                state_df = state_df.rename(
                    columns={"ship_state": "state", "quantity": "total_orders"}
                )
            top_states = state_df.where(pd.notnull(state_df), None).to_dict("records")
        except Exception:
            top_states = []

        # --- 5. Load Unpaid Orders ---
        try:
            unpaid_orders_df = pd.read_excel(output_path, sheet_name="Unpaid Orders")
            unpaid_orders = unpaid_orders_df.where(pd.notnull(unpaid_orders_df), None).to_dict("records")
        except Exception:
            unpaid_orders_df = pd.DataFrame()
            unpaid_orders = []

        # --- 6. Parse Summary Data ---
        summary_dict = {}
        for _, row in summary_df.iterrows():
            metric = str(row["Metric"]).strip()
            value = 0.0 if pd.isna(row["Value"]) else row["Value"]
            
            # Updated mapping for new revenue-based metrics
            mapping = {
                "Total Quantity Ordered": "total_quantity",
                "Total Revenue (All Orders)": "total_revenue",
                "Total Payment Received": "total_payment_received",
                "Total Pending Payment": "total_pending_payment",
                "Total Unpaid Order Quantity": "total_unpaid_quantity",
                "Total Cost": "total_cost",
                "Total Amz Fees": "total_amz_fees",
                "Total Return Quantity": "total_return_quantity",
                "Total Expenses": "total_expenses",
                "Net Profit": "net_profit",
                
                # Legacy mappings for backward compatibility
                "Total Quantity": "total_quantity",
                "Total Payment": "total_payment_received",
                "Total Unpaid Orders": "total_unpaid_quantity",
            }

            if metric in mapping:
                summary_dict[mapping[metric]] = value
            else:
                # Store dynamic expenses with a clean key
                clean_metric = metric.lower().replace(" ", "_").replace("-", "_")
                summary_dict[f"expense_{clean_metric}"] = value

        # Ensure all expected keys exist (for frontend compatibility)
        default_keys = {
            'total_quantity': 0.0,
            'total_revenue': 0.0,
            'total_payment_received': 0.0,
            'total_pending_payment': 0.0,
            'total_unpaid_quantity': 0.0,
            'total_cost': 0.0,
            'total_amz_fees': 0.0,
            'total_return_quantity': 0.0,
            'total_expenses': 0.0,
            'net_profit': 0.0,
        }
        
        for key, default_value in default_keys.items():
            if key not in summary_dict:
                summary_dict[key] = default_value

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