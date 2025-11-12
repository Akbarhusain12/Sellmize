import pandas as pd
import re
from datetime import datetime
import warnings
import os
import chardet

# Suppress warnings
warnings.filterwarnings('ignore')


def _clean_and_sum(value):
    """Helper function to clean and sum monetary values."""
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
    # --- Detect encoding automatically ---
    with open(txt_file_path, 'rb') as f:
        raw_data = f.read(50000)  # first 50KB for detection
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding'] or 'utf-8'
        confidence = result.get('confidence', 0)
    print(f"🔍 Detected encoding for {os.path.basename(txt_file_path)}: {detected_encoding} (confidence={confidence:.2f})")

    # --- Detect delimiter ---
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

    # --- Read file with safe encoding ---
    try:
        df = pd.read_csv(
            txt_file_path,
            sep=sep,
            engine='python',
            encoding=detected_encoding,
            on_bad_lines='skip'
        )
    except Exception as e:
        # Try fallback encodings
        for fallback in ['utf-16', 'cp1252', 'latin1']:
            try:
                df = pd.read_csv(
                    txt_file_path,
                    sep=sep,
                    engine='python',
                    encoding=fallback,
                    on_bad_lines='skip'
                )
                print(f"⚙️ Fallback to encoding: {fallback}")
                break
            except Exception:
                continue
        else:
            raise Exception(f"❌ Could not read {txt_file_path} with any encoding: {e}")

    # --- Validate ---
    if df.empty:
        raise Exception(f"❌ Conversion failed: {txt_file_path} produced empty DataFrame (encoding={detected_encoding}).")

    # --- Save as Excel ---
    base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    excel_path = os.path.join(output_folder, f"{base_name}.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"📄 Converted {ext.upper()} → Excel: {excel_path} | {df.shape[0]} rows, {df.shape[1]} columns (encoding={detected_encoding})")

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


def process_data(order_files, payment_files, return_files, cost_price_file,
                 dynamic_expenses, output_folder,
                 start_date=None, end_date=None):


    # --- 1. Load Data ---
    try:
        # --- 1. Load Data (Auto-handle TXT → Excel conversion) ---
        converted_order_files = []
        for file in order_files:
            ext = os.path.splitext(file)[-1].lower()
            if ext == '.txt':
                excel_path = convert_txt_to_excel(file, output_folder)
                converted_order_files.append(excel_path)
            else:
                converted_order_files.append(file)

        # Load all orders (Excel format only now)
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

        converted_return_files = []
        for file in return_files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in ['.txt', '.tsv']:
                excel_path = convert_txt_to_excel(file, output_folder)
                converted_return_files.append(excel_path)
            else:
                converted_return_files.append(file)

        # Load returns and cost price (now all Excel)
        Return = pd.concat([pd.read_excel(file) for file in converted_return_files], ignore_index=True)
        Cost_price = pd.read_excel(cost_price_file)
        

    except Exception as e:
        raise Exception(f"❌ Error reading input files: {e}")


    # --- 2. Find columns with flexible matching ---
    
    # Find all required columns from the orders file at the start
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
        if original_name: # Only add if the column was found
            col_rename_map[original_name] = clean_name
            final_col_list.append(original_name)

    # Filter orders and rename columns in one step
    filtered_orders = orders[
        (orders[quantity_col] > 0) & (orders[item_price_col].notna())
    ][final_col_list].copy()
    
    filtered_orders.rename(columns=col_rename_map, inplace=True)
    
    # Ensure all expected columns exist, even if blank (for consistency)
    for col in ['amazon-order-id', 'sku', 'quantity', 'item_price', 'order_date', 'ship_state']:
        if col not in filtered_orders.columns:
            filtered_orders[col] = None
            
    if 'order_date' in filtered_orders.columns and filtered_orders['order_date'].notna().any():
        try:
            # Convert to datetime safely
            filtered_orders['order_date'] = pd.to_datetime(filtered_orders['order_date'], errors='coerce').dt.date


            # Apply filters if provided
            if start_date:
                start_dt = pd.to_datetime(start_date).date()
                filtered_orders = filtered_orders[filtered_orders['order_date'] >= start_dt]

            if end_date:
                end_dt = pd.to_datetime(end_date).date()
                filtered_orders = filtered_orders[filtered_orders['order_date'] <= end_dt]


            print(f"📅 Date filtering applied: {len(filtered_orders)} records remain "
                  f"from {start_date or 'start'} to {end_date or 'end'}.")

        except Exception as e:
            print(f"⚠️ Date filtering skipped due to error: {e}")
    else:
        print("⚠️ No valid 'order_date' column found for date filtering.")
    
    # --- Apply date filtering to Return file as well ---
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

            print(f"🔄 Return file filtered by {return_date_col}: {len(Return)} records remain.")
        except Exception as e:
            print(f"⚠️ Could not filter Return file by date ({return_date_col}): {e}")
    else:
        print("⚠️ No valid 'Return request date' column found or all values are missing in Return file.")



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
        print(f"Warning: Could not find return columns. Found: {Return.columns.tolist()}")
        top_10_returns = pd.DataFrame(columns=['Rank', 'sku', 'quantity', 'category'])
        
    # --- Return file columns ---
    return_order_id_col = find_column(Return, ['Order ID', 'order id', 'orderid', 'order-id'])
    return_type_col = find_column(Return, ['Return type', 'return type', 'returntype', 'return_type', 'status'])

    if not all([return_order_id_col, return_type_col]):
        raise Exception(f"❌ Missing required columns in return files. Found columns: {Return.columns.tolist()}")

    filtered_return = Return[[return_order_id_col, return_type_col]].copy()
    filtered_return.columns = ['Order ID', 'Return type']

    # --- Payment file columns ---
    payment_order_id_col = find_column(payment, ['order id', 'orderid', 'order-id', 'order_id'])
    payment_total_col = find_column(payment, ['total', 'amount', 'total amount', 'Total', 'net total'])

    if not all([payment_order_id_col, payment_total_col]):
        raise Exception(f"❌ Missing required columns in payment files. Found columns: {payment.columns.tolist()}")

    filtered_payment = payment[[payment_order_id_col, payment_total_col]].copy()
    filtered_payment.columns = ['order id', 'total']
    filtered_payment = filtered_payment.dropna(subset=['order id'])

    # --- 3. Process Payments ---
    filtered_payment['total'] = filtered_payment['total'].apply(_clean_and_sum)
    consolidated_payment = filtered_payment.groupby('order id', as_index=False)['total'].sum()

    # --- 4. Merge Orders and Payments (only matching SKUs should remain) ---
    merged_data = pd.merge(
        filtered_orders,
        consolidated_payment,
        left_on='amazon-order-id',
        right_on='order id',
        how='left'
    )

    unpaid_orders = merged_data[merged_data['total'].isna()].copy()

    # Clean and keep relevant columns for reporting
    unpaid_orders = unpaid_orders[[
        'amazon-order-id', 'sku', 'quantity', 'item_price', 'order_date', 'ship_state'
    ]].copy()

    unpaid_orders.rename(columns={
        'amazon-order-id': 'Order ID',
        'sku': 'SKU',
        'quantity': 'Quantity',
        'item_price': 'Item Price',
        'order_date': 'Order Date',
        'ship_state': 'Ship State'
    }, inplace=True)

    unpaid_orders = unpaid_orders.sort_values(by='Order Date', ascending=False).reset_index(drop=True)
    print(f"⚠️ Found {len(unpaid_orders)} unpaid orders (present in Orders but missing in Payments).")
    total_unpaid_orders = unpaid_orders['Quantity'].sum()
    

    # Keep only rows where payment total exists (i.e., payment matched)
    merged_data = merged_data[merged_data['total'].notna()]

    # Optional: If multiple SKUs share order IDs, double-check SKU consistency
    # (We assume SKU and order id should both exist together)
    merged_data['sku'] = merged_data['sku'].astype(str).str.strip().str.upper()
    consolidated_sku_orders = filtered_orders[['amazon-order-id', 'sku']].copy()
    consolidated_sku_orders['sku'] = consolidated_sku_orders['sku'].astype(str).str.strip().str.upper()

    # Keep only SKUs that exist in both orders and payments
    merged_data = merged_data.merge(
        consolidated_sku_orders[['amazon-order-id', 'sku']],
        on=['amazon-order-id', 'sku'],
        how='inner'
    )

    
    # Define the columns we want to keep after the merge
    keep_cols = ['amazon-order-id', 'sku', 'quantity', 'total', 'item_price', 'order_date', 'ship_state']
    # Filter to only keep columns that actually exist in merged_data
    final_keep_cols = [col for col in keep_cols if col in merged_data.columns]
    merged_data = merged_data[final_keep_cols]


    # --- 5. Map Cost Price (only for SKUs present in both Order & Payment files) ---

    merged_data['sku'] = merged_data['sku'].astype(str).str.strip().str.upper()

    # Identify cost-related columns
    cost_sku_col = find_column(Cost_price, ['SKU', 'sku', 'product-sku', 'product sku'])
    cost_price_col = find_column(Cost_price, ['Product Cost', 'product cost', 'cost', 'price'])

    if not all([cost_sku_col, cost_price_col]):
        raise Exception(f"❌ Missing required columns in cost price file. Found columns: {Cost_price.columns.tolist()}")

    # Standardize SKU in cost file
    Cost_price[cost_sku_col] = Cost_price[cost_sku_col].astype(str).str.strip().str.upper()

    # ✅ Only use SKUs that appear in merged_data (i.e., Orders + Payments)
    paid_skus = set(merged_data['sku'].unique())
    Cost_price_filtered = Cost_price[Cost_price[cost_sku_col].isin(paid_skus)]

    # Create mapping only for those SKUs
    cost_price_dict = Cost_price_filtered.set_index(cost_sku_col)[cost_price_col].to_dict()

    # Map cost price only for paid SKUs
    merged_data['Product Cost'] = merged_data['sku'].map(cost_price_dict)

    # Calculate total cost only where both SKU and payment exist
    merged_data['Total Cost'] = merged_data['quantity'] * merged_data['Product Cost']

    # Optional: Warn if any SKU in merged_data still has no cost info
    # missing_cost_skus = merged_data[merged_data['Product Cost'].isna()]['sku'].unique()
    # if len(missing_cost_skus) > 0:
    #     print(f"⚠️ The following SKUs have payment but missing cost price: {missing_cost_skus}")


    # --- TOP 10 SKUs by Quantity Analysis ---
    top_10_data = merged_data[['sku', 'quantity']].copy()

    # --- TOP 10 STATES by Quantity Analysis ---
    if 'ship_state' in merged_data.columns and merged_data['ship_state'].notna().any():
        top_10_states_data = merged_data[['ship_state', 'quantity']].copy()
        top_10_states_data['ship_state'] = top_10_states_data['ship_state'].astype(str).str.strip().str.title()
        top_10_states_data['quantity'] = pd.to_numeric(top_10_states_data['quantity'], errors='coerce')
        top_10_states_data = top_10_states_data.dropna(subset=['quantity'])
        
        # Group by state and sum quantities
        top_10_states_grouped = top_10_states_data.groupby('ship_state', as_index=False)['quantity'].sum()
        top_10_states = top_10_states_grouped.sort_values('quantity', ascending=False).head(10).reset_index(drop=True)
        top_10_states.insert(0, 'Rank', range(1, len(top_10_states) + 1))
    else:
        print("⚠️ No 'ship_state' column found or all values are missing.")
        top_10_states = pd.DataFrame(columns=['Rank', 'ship_state', 'quantity'])

    # CRITICAL: Standardize SKU BEFORE grouping
    top_10_data['sku'] = top_10_data['sku'].astype(str).str.strip().str.upper()
    top_10_data['quantity'] = pd.to_numeric(top_10_data['quantity'], errors='coerce')
    top_10_data = top_10_data.dropna(subset=['quantity'])

    top_10_grouped = top_10_data.groupby('sku', as_index=False)['quantity'].sum()
    top_10_skus = top_10_grouped.sort_values('quantity', ascending=False).head(10).reset_index(drop=True)
    top_10_skus.insert(0, 'Rank', range(1, len(top_10_skus) + 1))

    # --- 6. Merge Return Status ---
    filtered_return['Order ID'] = filtered_return['Order ID'].astype(str)
    filtered_return = filtered_return.drop_duplicates(subset=['Order ID'], keep='last')


    merged_data = pd.merge(
        merged_data,
        filtered_return[['Order ID', 'Return type']],
        left_on='amazon-order-id',
        right_on='Order ID',
        how='left'
    )

    merged_data.rename(columns={'Return type': 'Status'}, inplace=True)
    merged_data.drop(columns=['Order ID'], inplace=True, errors='ignore')


    amz_fees_col = find_column(Cost_price, ['Amz Fees', 'amz fees', 'amazon fees', 'fees'])
    if amz_fees_col:
        amz_fees_dict = Cost_price.set_index(cost_sku_col)[amz_fees_col].to_dict()
        merged_data['Amz Fees'] = merged_data['sku'].map(amz_fees_dict)
    else:
        merged_data['Amz Fees'] = 0

    # --- 8. Calculate Summary ---
    filtered_na = merged_data[merged_data['Status'].isna()]
    total_quantity = filtered_na['quantity'].sum()
    total_payment = filtered_na['total'].sum()
    total_cost = filtered_na['Total Cost'].sum()



    # ✅ Calculate total returned quantity directly from Return file
    return_quantity_col = find_column(Return, ['Return quantity', 'return quantity', 'quantity', 'qty', 'returned qty'])
    if return_quantity_col:
        Return[return_quantity_col] = pd.to_numeric(Return[return_quantity_col], errors='coerce')
        total_return_quantity = Return[return_quantity_col].sum()
    else:
        total_return_quantity = 0


    total_amz_fees = merged_data['Amz Fees'].sum()

    # --- 9. Dynamic Expenses ---
    if not isinstance(dynamic_expenses, dict):
        raise Exception("❌ 'dynamic_expenses' must be a dictionary, e.g. {'Ad Cost': 1000, 'Office Rent': 2000}")

    total_expense = sum(dynamic_expenses.values())
    net_profit = total_payment - (total_cost + total_expense)

    # --- 10. Prepare dynamic summary table ---
    summary_rows = [
        ('Total Quantity', total_quantity),
        ('Total Payment', total_payment),
        ('Total Cost', total_cost),
        ('Total Amz Fees', total_amz_fees),
        ('Total Return Quantity', total_return_quantity),
        ('Total Unpaid Orders', total_unpaid_orders)
    ]


    # Add dynamic user expenses
    for name, value in dynamic_expenses.items():
        summary_rows.append((name, value))

    # Add final Net Profit
    summary_rows.append(('Net Profit', net_profit))

    summary_data = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])

    # --- 11. Reorder Columns ---
    cols = [
        'amazon-order-id', 'order_date', 'sku', 
        'quantity', 'total', 'Product Cost', 'Total Cost', 
        'Status', 'ship_state', 'Amz Fees'
    ]
    final_cols = [col for col in cols if col in merged_data.columns]
    merged_data = merged_data[final_cols]
    
    print("📈 Columns successfully ordered.")

    # --- 12. Save to Excel ---
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


    print(f"✅ Processing complete. File saved at: {output_path}")
    return file_name


