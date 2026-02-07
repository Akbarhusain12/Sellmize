import pandas as pd
import re
from datetime import datetime
import warnings
import os
import chardet
import logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# SECTION 1: CORE UTILITY FUNCTIONS
# --------------------------------------------------------------------------

def _clean_and_sum(value):
    """
    Helper function to clean and sum monetary values from strings.
    Handles currency symbols, commas, and multiple values in one string.
    """
    if isinstance(value, str):
        parts = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', value)
        cleaned_numbers = [float(part.replace(',', '')) for part in parts]
        return sum(cleaned_numbers)
    return value


def convert_txt_to_excel(txt_file_path, output_folder):
    """
    Converts a .txt, .tsv, or .csv file (comma, tab, or pipe-separated) to Excel (.xlsx)
    and returns the new Excel file path.
    Handles UTF-8, UTF-16, CP1252 automatically.
    """
    
    # --- 1. Detect encoding automatically ---
    with open(txt_file_path, 'rb') as f:
        raw_data = f.read(50000)  # first 50KB for detection
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding'] or 'utf-8'
        confidence = result.get('confidence', 0)
    logger.warning(f"🔍 Detected encoding for {os.path.basename(txt_file_path)}: {detected_encoding} (confidence={confidence:.2f})")

    # --- 2. Detect delimiter ---
    try:
        with open(txt_file_path, 'r', encoding=detected_encoding, errors='ignore') as f:
            sample = f.read(2048)
    except Exception:
        sample = ""

    ext = os.path.splitext(txt_file_path)[-1].lower()
    if ext == '.tsv' or '\t' in sample or ' \t' in sample:
        sep = '\t'
    elif '|' in sample:
        sep = '|'
    else:
        sep = ','

    # --- 3. Read file with safe encoding ---
    try:
        df = pd.read_csv(
            txt_file_path,
            sep=sep,
            engine='python',
            encoding=detected_encoding,
            on_bad_lines='skip'
        )
    except Exception as e:
        # Try fallback encodings if chardet fails
        for fallback in ['utf-16', 'cp1252', 'latin1']:
            try:
                df = pd.read_csv(
                    txt_file_path,
                    sep=sep,
                    engine='python',
                    encoding=fallback,
                    on_bad_lines='skip'
                )
                logger.warning(f"⚙️ Fallback to encoding: {fallback}")
                break
            except Exception:
                continue
        else:
            raise Exception(f"❌ Could not read {txt_file_path} with any encoding: {e}")

    # --- 4. Validate ---
    if df.empty:
        raise Exception(f"❌ Conversion failed: {txt_file_path} produced empty DataFrame (encoding={detected_encoding}).")

    # --- 5. Save as Excel ---
    base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    excel_path = os.path.join(output_folder, f"{base_name}.xlsx")
    df.to_excel(excel_path, index=False)
    logger.warning(f"📄 Converted {ext.upper()} → Excel: {excel_path} | {df.shape[0]} rows, {df.shape[1]} columns (encoding={detected_encoding})")

    return excel_path


def find_column(df, possible_names):
    """
    Find a column in dataframe by flexible name matching (case-insensitive + partial).
    """
    cols_lower = {col.lower().strip(): col for col in df.columns}
    for name in possible_names:
        name_lower = name.lower().strip()
        # Exact match
        if name_lower in cols_lower:
            return cols_lower[name_lower]
        # Partial match
        for col_lower, original in cols_lower.items():
            if name_lower in col_lower or col_lower in name_lower:
                return original
    return None

# --------------------------------------------------------------------------
# SECTION 2: MAIN DATA PROCESSING ORCHESTRATOR
# --------------------------------------------------------------------------

def process_data(order_files, payment_files, return_files, cost_price_file,
                 dynamic_expenses, output_folder,
                 start_date=None, end_date=None):
    """
    Main data processing pipeline - REVENUE-BASED CALCULATION.
    Calculates based on ALL orders, tracks payments separately.
    """

    # --- 1. Load Data (with TXT-to-Excel conversion) ---
    logger.warning("--- 1. Loading Data ---")
    try:
        # Convert text-based order files to Excel
        converted_order_files = []
        for file in order_files:
            ext = os.path.splitext(file)[-1].lower()
            if ext == '.txt':
                excel_path = convert_txt_to_excel(file, output_folder)
                converted_order_files.append(excel_path)
            else:
                converted_order_files.append(file)
        orders = pd.concat([pd.read_excel(file) for file in converted_order_files], ignore_index=True)

        # Load payments
        payment_dfs = []
        for file in payment_files:
            try:
                df = pd.read_csv(file, skiprows=11)
            except Exception:
                df = pd.read_csv(file)
            payment_dfs.append(df)
        payment = pd.concat(payment_dfs, ignore_index=True)

        # Convert text-based return files to Excel
        converted_return_files = []
        for file in return_files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in ['.txt', '.tsv']:
                excel_path = convert_txt_to_excel(file, output_folder)
                converted_return_files.append(excel_path)
            else:
                converted_return_files.append(file)
        Return = pd.concat([pd.read_excel(file) for file in converted_return_files], ignore_index=True)
        
        # Load cost price
        Cost_price = pd.read_excel(cost_price_file)

    except Exception as e:
        raise Exception(f"❌ Error reading input files: {e}")

    # --- 2. Find Columns and Filter by Date ---
    logger.warning("--- 2. Finding Columns & Filtering Dates ---")
    
    # Find all required columns from the orders file
    order_id_col = find_column(orders, ['amazon-order-id', 'order-id', 'order id', 'orderid', 'amazon order id'])
    sku_col = find_column(orders, ['sku', 'SKU', 'product-sku'])
    quantity_col = find_column(orders, ['quantity', 'qty', 'quantity-purchased', 'quantity purchased'])
    item_price_col = find_column(orders, ['item-price', 'item price', 'price', 'itemprice'])
    purchase_date_col = find_column(orders, ['purchase-date', 'purchase date', 'order date', 'order_date'])
    ship_state_col = find_column(orders, ['ship-state', 'ship state', 'shipstate', 'state', 'shipping-state', 'order state'])

    # Build a list of columns we absolutely need
    required_cols = {
        'order_id': order_id_col,
        'sku': sku_col,
        'quantity': quantity_col,
        'item_price': item_price_col
    }

    # Check for missing required columns
    missing = [name for name, col in required_cols.items() if col is None]
    if missing:
        raise Exception(
            f"❌ Missing required columns in order files: {', '.join(missing)}.\n"
            f"Found columns: {orders.columns.tolist()}\n"
            f"Expected something similar to: ['order id', 'sku', 'quantity', 'item price']"
        )
    
    # Build a list of all columns to extract (required + optional)
    extract_cols = {
        'amazon-order-id': order_id_col,
        'sku': sku_col,
        'quantity': quantity_col,
        'item_price': item_price_col,
        'order_date': purchase_date_col,
        'ship_state': ship_state_col
    }

    # Create the rename map and the final list of columns to pull
    col_rename_map = {}
    final_col_list = []
    for clean_name, original_name in extract_cols.items():
        if original_name:
            col_rename_map[original_name] = clean_name
            final_col_list.append(original_name)

    # Filter orders and rename columns
    filtered_orders = orders[
        (orders[quantity_col] > 0) & (orders[item_price_col].notna())
    ][final_col_list].copy()
    
    filtered_orders.rename(columns=col_rename_map, inplace=True)
    
    # Ensure all expected columns exist
    for col in ['amazon-order-id', 'sku', 'quantity', 'item_price', 'order_date', 'ship_state']:
        if col not in filtered_orders.columns:
            filtered_orders[col] = None
            
    # --- Date Filtering (Orders) ---
    if 'order_date' in filtered_orders.columns and filtered_orders['order_date'].notna().any():
        try:
            temp_date = pd.to_datetime(filtered_orders['order_date'], errors='coerce')
            if temp_date.dt.tz is not None:
                temp_date = temp_date.dt.tz_localize(None)
            
            filtered_orders['order_date'] = temp_date.dt.date

            if start_date:
                start_dt = pd.to_datetime(start_date).date()
                filtered_orders = filtered_orders[filtered_orders['order_date'] >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date).date()
                filtered_orders = filtered_orders[filtered_orders['order_date'] <= end_dt]

            logger.warning(f"📅 Date filtering applied: {len(filtered_orders)} records remain.")
        except Exception as e:
            logger.warning(f"⚠️ Date filtering skipped: {e}")
    else:
        logger.warning("⚠️ No valid 'order_date' column found for date filtering.")
    
    # --- Date Filtering (Returns) ---
    return_date_col = find_column(Return, [
        'Return request date', 'return-request-date', 'return date', 'requested date'
    ])

    if return_date_col and Return[return_date_col].notna().any():
        try:
            Return[return_date_col] = pd.to_datetime(Return[return_date_col], errors='coerce').dt.date

            if start_date:
                start_dt = pd.to_datetime(start_date).date()
                Return = Return[Return[return_date_col] >= start_dt]

            if end_date:
                end_dt = pd.to_datetime(end_date).date()
                Return = Return[Return[return_date_col] <= end_dt]

            logger.warning(f"🔄 Return file filtered by {return_date_col}: {len(Return)} records remain.")
        except Exception as e:
            logger.warning(f"⚠️ Could not filter Return file by date ({return_date_col}): {e}")
    else:
        logger.warning("⚠️ No valid 'Return request date' column found in Return file.")

    # ================== ANALYTICS SNAPSHOT ==================
    orders_snapshot = filtered_orders.copy()
    orders_snapshot['order_date'] = pd.to_datetime(
        orders_snapshot['order_date'], errors='coerce'
    )

    orders_snapshot = orders_snapshot.dropna(subset=['order_date'])
    orders_snapshot['weekday'] = orders_snapshot['order_date'].dt.dayofweek

    orders_by_weekday = (
        orders_snapshot['weekday']
        .value_counts()
        .sort_index()
        .reset_index()
    )
    orders_by_weekday.columns = ['weekday', 'order_count']

    weekday_map = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }
    orders_by_weekday['weekday'] = orders_by_weekday['weekday'].map(weekday_map)

    daily_orders = (
        orders_snapshot
        .set_index('order_date')
        .resample('D')
        .size()
        .reset_index(name='order_count')
    )
    
    orders_snapshot['item_price'] = pd.to_numeric(
        orders_snapshot['item_price'], errors='coerce'
    ).fillna(0)

    # -------------------------------------------------------------
    # MODIFIED: Revenue calculation (Item Price ONLY, no quantity)
    # -------------------------------------------------------------
    orders_snapshot['revenue'] = orders_snapshot['item_price']

    daily_revenue = (
        orders_snapshot
        .set_index('order_date')['revenue']
        .resample('D')
        .sum()
        .reset_index()
    )
    
    return_reason_col = find_column(Return, [
        'Return reason', 'return reason', 'reason'
    ])

    if return_reason_col:
        return_reasons = (
            Return[return_reason_col]
            .value_counts()
            .reset_index()
        )
        return_reasons.columns = ['reason', 'count']
    else:
        return_reasons = pd.DataFrame(columns=['reason', 'count'])

    # --- 3. Pre-Merge Analysis (Top 10s) ---
    logger.warning("--- 3. Running Pre-Merge Analysis ---")
    
    # --- TOP 10 RETURNED SKUs Analysis ---
    return_sku_col = find_column(Return, ['Merchant SKU', 'merchant sku', 'sku', 'SKU', 'product-sku'])
    return_quantity_col = find_column(Return, ['Return quantity', 'return quantity', 'quantity', 'qty', 'returned qty'])
    return_category_col = find_column(Return, ['Category', 'category', 'return category', 'return type', 'reason'])

    if all([return_sku_col, return_quantity_col, return_category_col]):
        top_10_returns_data = Return[[return_sku_col, return_quantity_col, return_category_col]].copy()
        top_10_returns_data.columns = ['sku', 'quantity', 'category']
        top_10_returns_data['sku'] = top_10_returns_data['sku'].astype(str).str.strip().str.upper()
        top_10_returns_data['quantity'] = pd.to_numeric(top_10_returns_data['quantity'], errors='coerce')
        top_10_returns_data = top_10_returns_data.dropna(subset=['quantity', 'sku'])
        top_10_returns_grouped = top_10_returns_data.groupby(['sku', 'category'], as_index=False)['quantity'].sum()
        top_10_returns = top_10_returns_grouped.sort_values('quantity', ascending=False).head(10).reset_index(drop=True)
        top_10_returns.insert(0, 'Rank', range(1, len(top_10_returns) + 1))
    else:
        logger.warning(f"Warning: Could not find return columns. Found: {Return.columns.tolist()}")
        top_10_returns = pd.DataFrame(columns=['Rank', 'sku', 'quantity', 'category'])

    # --- 4. Process Payments ---
    logger.warning("--- 4. Processing Payments ---")

    payment_order_id_col = find_column(payment, ['order id', 'orderid', 'order-id', 'order_id'])
    payment_total_col = find_column(payment, ['total', 'amount', 'total amount', 'Total', 'net total'])

    if not all([payment_order_id_col, payment_total_col]):
        raise Exception(f"❌ Missing required columns in payment files. Found columns: {payment.columns.tolist()}")

    filtered_payment = payment[[payment_order_id_col, payment_total_col]].copy()
    filtered_payment.columns = ['order id', 'total']
    filtered_payment = filtered_payment.dropna(subset=['order id'])
    filtered_payment['total'] = filtered_payment['total'].apply(_clean_and_sum)
    consolidated_payment = filtered_payment.groupby('order id', as_index=False)['total'].sum()

    # --- 5. Calculate Total Revenue (ALL ORDERS) ---
    logger.warning("--- 5. Calculating Total Revenue from ALL Orders ---")
    
    # Clean and prepare orders data
    filtered_orders['sku'] = filtered_orders['sku'].astype(str).str.strip().str.upper()
    filtered_orders['item_price'] = pd.to_numeric(filtered_orders['item_price'], errors='coerce').fillna(0)
    filtered_orders['quantity'] = pd.to_numeric(filtered_orders['quantity'], errors='coerce').fillna(0)
    
    # -------------------------------------------------------------
    # MODIFIED: Revenue calculation (Item Price ONLY, no quantity)
    # -------------------------------------------------------------
    filtered_orders['revenue'] = filtered_orders['item_price']

    # --- 6. Merge with Payments to Track Payment Status ---
    logger.warning("--- 6. Merging Orders with Payments ---")
    
    merged_data = pd.merge(
        filtered_orders,
        consolidated_payment,
        left_on='amazon-order-id',
        right_on='order id',
        how='left'
    )
    
    # Add payment status flag
    merged_data['payment_status'] = merged_data['total'].notna()
    merged_data['payment_received'] = merged_data['total'].fillna(0)
    
    # Clean up duplicate column
    merged_data.drop(columns=['order id'], inplace=True, errors='ignore')

    # --- 7. Identify Unpaid Orders ---
    logger.warning("--- 7. Identifying Unpaid Orders ---")
    
    unpaid_orders = merged_data[~merged_data['payment_status']].copy()
    unpaid_orders = unpaid_orders[
        ['amazon-order-id', 'sku', 'quantity', 'item_price', 'revenue', 'order_date', 'ship_state']
    ].copy()
    unpaid_orders.rename(columns={
        'amazon-order-id': 'Order ID',
        'sku': 'SKU',
        'quantity': 'Quantity',
        'item_price': 'Item Price',
        'revenue': 'Pending Revenue',
        'order_date': 'Order Date',
        'ship_state': 'Ship State'
    }, inplace=True)
    unpaid_orders = unpaid_orders.sort_values(by='Order Date', ascending=False).reset_index(drop=True)
    
    total_unpaid_quantity = unpaid_orders['Quantity'].sum()
    total_pending_revenue = unpaid_orders['Pending Revenue'].sum()
    
    logger.warning(f"⚠️ Found {len(unpaid_orders)} unpaid order lines:")
    logger.warning(f"   - Total Unpaid Quantity: {total_unpaid_quantity}")
    logger.warning(f"   - Total Pending Revenue: ₹{total_pending_revenue:,.2f}")

    # --- 8. Map Cost Price (on ALL data) ---
    logger.warning("--- 8. Mapping Cost Price ---")
    
    cost_sku_col = find_column(Cost_price, ['SKU', 'sku', 'product-sku', 'product sku'])
    cost_price_col = find_column(Cost_price, ['Product Cost', 'product cost', 'cost', 'price'])

    if not all([cost_sku_col, cost_price_col]):
        raise Exception(f"❌ Missing required columns in cost price file. Found columns: {Cost_price.columns.tolist()}")

    Cost_price[cost_sku_col] = Cost_price[cost_sku_col].astype(str).str.strip().str.upper()
    cost_price_dict = Cost_price.set_index(cost_sku_col)[cost_price_col].to_dict()

    merged_data['Product Cost'] = merged_data['sku'].map(cost_price_dict)
    merged_data['Total Cost'] = merged_data['quantity'] * merged_data['Product Cost']

    # Add Amz Fees
    amz_fees_col = find_column(Cost_price, ['Amz Fees', 'amz fees', 'amazon fees', 'fees'])
    if amz_fees_col:
        amz_fees_dict = Cost_price.set_index(cost_sku_col)[amz_fees_col].to_dict()
        merged_data['Amz Fees'] = merged_data['sku'].map(amz_fees_dict)
    else:
        merged_data['Amz Fees'] = 0

    # --- 9. Merge Returns Data ---
    logger.warning("--- 9. Merging Returns Data ---")
    
    return_order_id_col = find_column(Return, ['Order ID', 'order id', 'orderid', 'order-id'])
    return_type_col = find_column(Return, ['Return type', 'return type', 'returntype', 'return_type', 'Status'])
    raw_reason_col = find_column(Return, ['Return reason', 'reason', 'return-reason', 'detailed-reason'])

    if all([return_order_id_col, return_type_col]):
        return_lookup = Return[[return_order_id_col, return_type_col]].copy()
        return_lookup.columns = ['Order ID', 'Return type']
        
        if raw_reason_col:
            return_lookup['raw_return_reason'] = Return[raw_reason_col]
        else:
            return_lookup['raw_return_reason'] = "Unknown"
        
        return_lookup['Order ID'] = return_lookup['Order ID'].astype(str)
        return_lookup = return_lookup.drop_duplicates(subset=['Order ID'], keep='last')
        
        merged_data = pd.merge(
            merged_data,
            return_lookup,
            left_on='amazon-order-id',
            right_on='Order ID',
            how='left'
        )
        merged_data.drop(columns=['Order ID'], inplace=True, errors='ignore')
    else:
        merged_data['Return type'] = None
        merged_data['raw_return_reason'] = None

    merged_data.rename(columns={'Return type': 'Status'}, inplace=True)

    # --- 10. Post-Merge Analysis ---
    logger.warning("--- 10. Running Post-Merge Analysis ---")

    # TOP 10 STATES by Quantity
    if 'ship_state' in merged_data.columns and merged_data['ship_state'].notna().any():
        top_10_states_data = merged_data[['ship_state', 'quantity']].copy()
        top_10_states_data['ship_state'] = top_10_states_data['ship_state'].astype(str).str.strip().str.title()
        top_10_states_data['quantity'] = pd.to_numeric(top_10_states_data['quantity'], errors='coerce')
        top_10_states_data = top_10_states_data.dropna(subset=['quantity'])
        top_10_states_grouped = top_10_states_data.groupby('ship_state', as_index=False)['quantity'].sum()
        top_10_states = top_10_states_grouped.sort_values('quantity', ascending=False).head(10).reset_index(drop=True)
        top_10_states.insert(0, 'Rank', range(1, len(top_10_states) + 1))
    else:
        top_10_states = pd.DataFrame(columns=['Rank', 'ship_state', 'quantity'])

    # TOP 10 SKUs by Quantity (ALL orders)
    top_10_data = merged_data[['sku', 'quantity']].copy()
    top_10_data['sku'] = top_10_data['sku'].astype(str).str.strip().str.upper()
    top_10_data['quantity'] = pd.to_numeric(top_10_data['quantity'], errors='coerce')
    top_10_data = top_10_data.dropna(subset=['quantity'])
    top_10_grouped = top_10_data.groupby('sku', as_index=False)['quantity'].sum()
    top_10_skus = top_10_grouped.sort_values('quantity', ascending=False).head(10).reset_index(drop=True)
    top_10_skus.insert(0, 'Rank', range(1, len(top_10_skus) + 1))

    # --- 11. Calculate Final Summary (REVENUE-BASED) ---
    logger.warning("--- 11. Calculating Final Summary (Revenue-Based) ---")
    
    # Filter out returned items for profit calculation
    status_col = next((c for c in merged_data.columns if c.lower() == 'status'), None)
    
    if status_col:
        non_returned = merged_data[merged_data[status_col].isna()]
    else:
        non_returned = merged_data.copy()

    # TOTAL CALCULATIONS (ALL ORDERS)
    total_quantity = merged_data['quantity'].sum()
    total_revenue = merged_data['revenue'].sum()
    
    # PAYMENT TRACKING
    total_payment_received = merged_data['payment_received'].sum()
    total_pending_payment = total_revenue - total_payment_received
    
    # COST CALCULATIONS (excluding returns)
    total_cost = non_returned['Total Cost'].sum()
    total_amz_fees = non_returned['Amz Fees'].sum()
    
    # RETURNS
    return_quantity_col = find_column(Return, ['Return quantity', 'return quantity', 'quantity', 'qty', 'returned qty'])
    if return_quantity_col:
        Return[return_quantity_col] = pd.to_numeric(Return[return_quantity_col], errors='coerce')
        total_return_quantity = Return[return_quantity_col].sum()
    else:
        total_return_quantity = 0

    # DYNAMIC EXPENSES
    if not isinstance(dynamic_expenses, dict):
        raise Exception("❌ 'dynamic_expenses' must be a dictionary")

    total_expense = sum(dynamic_expenses.values())
    
    # NET PROFIT CALCULATION (Revenue - Costs - Expenses)
    net_profit = total_revenue - (total_cost + total_amz_fees + total_expense)

    # --- 12. Prepare Summary Table ---
    summary_rows = [
        ('Total Quantity Ordered', total_quantity),
        ('Total Revenue (All Orders)', total_revenue),
        ('Total Payment Received', total_payment_received),
        ('Total Pending Payment', total_pending_payment),
        ('Total Unpaid Order Quantity', total_unpaid_quantity),
        ('Total Cost', total_cost),
        ('Total Amz Fees', total_amz_fees),
        ('Total Return Quantity', total_return_quantity),
    ]

    # Add dynamic expenses
    for name, value in dynamic_expenses.items():
        summary_rows.append((name, value))

    # Add final metrics
    summary_rows.append(('Total Expenses', total_expense))
    summary_rows.append(('Net Profit', net_profit))
    
    summary_data = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])

    # --- 13. Reorder Columns for Final Report ---
    logger.warning("--- 13. Preparing Final Report ---")
    
    cols = [
        'amazon-order-id', 'order_date', 'sku', 'quantity',
        'item_price', 'revenue', 'payment_status', 'payment_received',
        'Product Cost', 'Total Cost', 'Amz Fees',
        'Status', 'raw_return_reason', 'ship_state'
    ]

    final_cols = [col for col in cols if col in merged_data.columns]
    merged_data = merged_data[final_cols]

    # Rename for clarity
    merged_data.rename(columns={
        'payment_status': 'Is Paid',
        'payment_received': 'Payment Amount'
    }, inplace=True)

    # --- 14. Save Final Report ---
    logger.warning("--- 14. Saving Final Report ---")
    
    current_date = datetime.now().strftime('%d-%B-%Y-%H-%M')
    file_name = f"{current_date}.xlsx"
    output_path = os.path.join(output_folder, file_name)

    with pd.ExcelWriter(output_path) as writer:
        merged_data.to_excel(writer, sheet_name='Merged Data', index=False)
        summary_data.to_excel(writer, sheet_name='Summary', index=False)
        top_10_skus.to_excel(writer, sheet_name='Top 10 SKUs', index=False)
        top_10_returns.to_excel(writer, sheet_name='Top 10 Returns', index=False)
        top_10_states.to_excel(writer, sheet_name='Top 10 States', index=False)
        unpaid_orders.to_excel(writer, sheet_name='Unpaid Orders', index=False)
        orders_by_weekday.to_excel(writer, sheet_name='Orders_Weekday', index=False)
        daily_orders.to_excel(writer, sheet_name='Daily_Orders', index=False)
        daily_revenue.to_excel(writer, sheet_name='Daily_Revenue', index=False)
        return_reasons.to_excel(writer, sheet_name='Return_Reasons', index=False)

    logger.warning(f"✅ Processing complete. File saved at: {output_path}")
    logger.warning(f"📊 Summary:")
    logger.warning(f"   Total Revenue: ₹{total_revenue:,.2f}")
    logger.warning(f"   Payment Received: ₹{total_payment_received:,.2f}")
    logger.warning(f"   Pending Payment: ₹{total_pending_payment:,.2f}")
    logger.warning(f"   Net Profit: ₹{net_profit:,.2f}")
    
    return file_name