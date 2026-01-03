import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify, send_file, session
from werkzeug.utils import secure_filename
from content_generation import generate_product_content
from agents.MarketLens import UnifiedMarketAgent
from config import Config
from ecommerce_analyzer import process_data
import pandas as pd
import chardet
import google.generativeai as genai
import json
import asyncio
import re
from dotenv import load_dotenv
from datetime import datetime
from sku_health import analyze_sku_health
import logging
from datetime import timedelta
import shutil
import tempfile
from DB.db import database, init_app
from DB.models import *
from DB.db_writer import save_full_analysis
from sqlalchemy import func, desc

# Configure logging ONCE at the top
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Using gemini-2.0-flash as per your latest successful check
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

# Initialize Flask App
app = Flask(__name__)

# 🔥 CRITICAL: Proper secret key configuration
SECRET_KEY = os.getenv('FLASK_SECRET')
if not SECRET_KEY:
    SECRET_KEY = os.urandom(32).hex()
    logger.warning("⚠️ No FLASK_SECRET found, using random key")
else:
    logger.info("✅ Using FLASK_SECRET from environment")

app.secret_key = SECRET_KEY
app.config.from_object(Config)

init_app(app)

# ============================================================================
# 🔥 SESSION CONFIGURATION
# ============================================================================

app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
app.config["SESSION_COOKIE_NAME"] = "sellmate_session"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Secure cookies only in production
if os.getenv('FLASK_ENV') == 'production' or os.getenv('RENDER'):
    app.config["SESSION_COOKIE_SECURE"] = True
else:
    app.config["SESSION_COOKIE_SECURE"] = False

# Make sessions permanent by default
@app.before_request
def make_session_permanent():
    session.permanent = False

logger.info("✅ Session configuration complete")

# ============================================================================
# HELPER FUNCTION TO LOAD DATA FROM EXCEL
# ============================================================================

def load_analysis_from_file(filename):
    """
    Load analysis data from the saved Excel file.
    """
    try:
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(output_path):
            logger.error(f"File not found: {output_path}")
            return None
        
        logger.info(f"📂 Loading data from file: {filename}")
        
        # 1. Read Summary
        summary_df = pd.read_excel(output_path, sheet_name='Summary')
        
        # 2. Read Top 10 SKUs
        try:
            top_10_df = pd.read_excel(output_path, sheet_name='Top 10 SKUs')
            top_10_list = top_10_df.to_dict('records')
        except Exception:
            top_10_list = []

        # 3. Read Top 10 Returns
        try:
            top_10_returns_df = pd.read_excel(output_path, sheet_name='Top 10 Returns')
            top_10_returns_list = top_10_returns_df.to_dict('records')
        except Exception:
            top_10_returns_list = []

        # 4.  NEW: Read Top 10 States
        try:
            top_states_df = pd.read_excel(output_path, sheet_name='Top 10 States')
            # Rename columns to match frontend expectations
            if 'ship_state' in top_states_df.columns:
                top_states_df = top_states_df.rename(columns={'ship_state': 'state', 'quantity': 'total_orders'})
            top_states_list = top_states_df.to_dict('records')
        except Exception as e:
            logger.warning(f"Could not read Top 10 States: {e}")
            top_states_list = []

        # 5.  NEW: Read Unpaid Orders
        try:
            unpaid_orders_df = pd.read_excel(output_path, sheet_name='Unpaid Orders')
            unpaid_orders_list = unpaid_orders_df.to_dict('records')
        except Exception as e:
            logger.warning(f"Could not read Unpaid Orders: {e}")
            unpaid_orders_list = []
            unpaid_orders_df = pd.DataFrame() # Create empty for SKU health check

        # 6. NEW: Re-Run SKU Health Analysis
        # We need to re-run the logic because health scores aren't saved as raw data
        sku_health_rows = []
        try:
            merged_data_df = pd.read_excel(output_path, sheet_name='Merged Data')
            # Re-run the analysis function you imported
            health_result = analyze_sku_health(
                merged_data=merged_data_df,
                returns=top_10_returns_df if 'top_10_returns_df' in locals() else pd.DataFrame(),
                unpaid_orders=unpaid_orders_df if 'unpaid_orders_df' in locals() else pd.DataFrame(),
                min_orders=1
            )
            
            # Format results exactly like in the analyze() route
            scored_df = health_result.scored_df
            scored_df['health_score_pred'] = scored_df['health_score_pred'].round(2)
            scored_df['proxy_label'] = scored_df['proxy_label'].round(2)
            
            def get_health_rating(score):
                if score >= 75: return "Excellent"
                elif score >= 60: return "Good"
                elif score >= 40: return "Fair"
                else: return "Poor"
            scored_df['health_rating'] = scored_df['health_score_pred'].apply(get_health_rating)
            
            # Rename for UI
            scored_df = scored_df.rename(columns={
                'SKU': 'SKU', 'health_score_pred': 'Score', 'proxy_label': 'Rating',
                'cluster': 'Segment', 'anomaly': 'Anomaly', 'health_rating': 'Health',
                'key_issues': 'Key_Issues'
            })
            
            display_cols = ['SKU', 'Score', 'Health', 'Key_Issues']
            if 'Rating' in scored_df.columns: display_cols.append('Rating')
            if 'Anomaly' in scored_df.columns:
                scored_df['Anomaly'] = scored_df['Anomaly'].map({True: 'Yes', False: 'No'})
                display_cols.append('Anomaly')
                
            sku_health_rows = scored_df[display_cols].to_dict('records')
        except Exception as e:
            logger.warning(f"Could not re-calculate SKU Health: {e}")

        # Process Summary Data (Existing logic)
        summary_dict = {}
        for _, row in summary_df.iterrows():
            metric = str(row['Metric']).strip()
            value = row['Value']
            try: value = float(value)
            except: value = 0.0
            
            key_mapping = {
                'Total Quantity': 'total_quantity', 'Total Return Quantity': 'total_return_quantity',
                'Total Unpaid Orders': 'total_unpaid_orders', 'Total Payment': 'total_payment',
                'Total Cost': 'total_cost', 'Total Amz Fees': 'total_amz_fees', 'Net Profit': 'net_profit'
            }
            if metric in key_mapping: summary_dict[key_mapping[metric]] = value
            elif metric.lower() not in ['total quantity', 'total payment', 'total cost', 'net profit']:
                clean_key = "expense_" + metric.lower().replace(" ", "_")
                summary_dict[clean_key] = value
        
        logger.info(f"✅ Loaded all data successfully")
        
        return {
            'summary_data': summary_dict,
            'top_10_skus': top_10_list,
            'top_10_returns': top_10_returns_list,
            'top_states': top_states_list,       # Added
            'unpaid_orders': unpaid_orders_list, # Added
            'sku_health': sku_health_rows        # Added
        }
        
    except Exception as e:
        logger.error(f"❌ Error loading file: {e}", exc_info=True)
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_uploaded_files(order_files, payment_files, return_files, cost_price_file):
    """Helper function to save uploaded files and return their paths."""
    upload_dir = app.config['UPLOAD_FOLDER']
    
    def save_files(files):
        paths = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                path = os.path.join(upload_dir, filename)
                file.save(path)
                paths.append(path)
        return paths

    order_paths = save_files(order_files)
    payment_paths = save_files(payment_files)
    return_paths = save_files(return_files)
    
    # Save single cost price file
    cost_price_path = ""
    if cost_price_file and cost_price_file.filename:
        filename = secure_filename(cost_price_file.filename)
        cost_price_path = os.path.join(upload_dir, filename)
        cost_price_file.save(cost_price_path)
    
    return order_paths, payment_paths, return_paths, cost_price_path


def convert_txt_to_excel(txt_file_path, output_folder):
    """
    Converts a .txt, .tsv, or .csv file to Excel (.xlsx).
    """
    # Detect encoding automatically
    with open(txt_file_path, 'rb') as f:
        raw_data = f.read(50000)
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding'] or 'utf-8'
        confidence = result.get('confidence', 0)
    logger.info(f"Detected encoding for {os.path.basename(txt_file_path)}: {detected_encoding} (confidence={confidence:.2f})")

    # Detect delimiter
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

    # Read file with safe encoding
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
                logger.info(f"Fallback to encoding: {fallback}")
                break
            except Exception:
                continue
        else:
            raise Exception(f"Could not read {txt_file_path} with any encoding: {e}")

    # Validate
    if df.empty:
        raise Exception(f"Conversion failed: {txt_file_path} produced empty DataFrame.")

    # Save as Excel
    base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    excel_path = os.path.join(output_folder, f"{base_name}.xlsx")
    df.to_excel(excel_path, index=False)
    logger.info(f"Converted {ext.upper()} → Excel: {excel_path} | {df.shape[0]} rows")

    return excel_path


# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


# ============================================================================
# MAIN ROUTES
# ============================================================================

@app.route('/')
def index():
    """
    Renders the Dashboard with optional month filtering.
    """
    from sqlalchemy import extract
    
    # --- 1. GET AVAILABLE MONTHS ---
    available_months = []
    try:
        # Get distinct year-month combinations from Analysis table
        months_query = db.session.query(
            extract('year', Analysis.start_date).label('year'),
            extract('month', Analysis.start_date).label('month')
        ).distinct().order_by(
            extract('year', Analysis.start_date).desc(),
            extract('month', Analysis.start_date).desc()
        ).all()
        
        for year, month in months_query:
            if year and month:
                date_obj = datetime(int(year), int(month), 1)
                available_months.append({
                    'value': f"{int(year)}-{int(month):02d}",
                    'label': date_obj.strftime('%B %Y')
                })
    except Exception as e:
        logger.warning(f"Could not fetch available months: {e}")

    # --- 2. GET SELECTED MONTH FROM URL ---
    selected_month = request.args.get('month', None)

    # --- 3. INITIALIZE CONTEXT ---
    context = {
        'data_exists': False,
        'revenue': '0.00',
        'total_orders': '0',
        'return_rate': '0.00',
        'net_profit': '0.00',
        'chart_sku_labels': [],
        'chart_sku_data': [],
        'chart_state_labels': [],
        'chart_state_data': [],
        'chart_trend_labels': [],
        'chart_trend_revenue': [],
        'chart_trend_profit': [],
        'chart_cost_labels': [],
        'chart_cost_data': [],
        'last_updated': 'Never',
        'available_months': available_months,
        'selected_month': selected_month or 'all'
    }

    # --- 4. FILTER ANALYSES BASED ON MONTH ---
    try:
        if selected_month and selected_month != 'all':
            # Parse YYYY-MM format
            try:
                year, month = map(int, selected_month.split('-'))
                analyses = Analysis.query.filter(
                    extract('year', Analysis.start_date) == year,
                    extract('month', Analysis.start_date) == month
                ).all()
                logger.info(f"📅 Filtered {len(analyses)} analyses for {selected_month}")
            except ValueError:
                logger.error(f"Invalid month format: {selected_month}")
                analyses = []
        else:
            # Show all data (default)
            analyses = Analysis.query.all()
            logger.info(f"📊 Loading all {len(analyses)} analyses")

        if not analyses:
            return render_template('index.html', **context)

        # --- 5. AGGREGATE DATA FROM FILTERED ANALYSES ---
        analysis_ids = [a.id for a in analyses]
        context['data_exists'] = True
        context['last_updated'] = analyses[-1].created_at.strftime('%B %d, %Y')

        # --- 5a. AGGREGATE SUMMARY METRICS ---
        metrics_sum = {}
        all_metrics = SummaryMetric.query.filter(
            SummaryMetric.analysis_id.in_(analysis_ids)
        ).all()
        
        for metric_obj in all_metrics:
            if metric_obj.metric not in metrics_sum:
                metrics_sum[metric_obj.metric] = 0
            metrics_sum[metric_obj.metric] += metric_obj.value

        # Calculate KPIs
        revenue = metrics_sum.get('total_payment', 0)
        net_profit = metrics_sum.get('net_profit', 0)
        total_qty = metrics_sum.get('total_quantity', 0)
        return_qty = metrics_sum.get('total_return_quantity', 0)

        context['revenue'] = f"{revenue:,.2f}"
        context['net_profit'] = f"{net_profit:,.2f}"
        context['total_orders'] = f"{int(total_qty):,}"
        context['return_rate'] = f"{(return_qty / total_qty * 100) if total_qty > 0 else 0:.2f}"

        # --- 5b. AGGREGATE TOP SKUs ---
        sku_data = db.session.query(
            TopSKU.sku,
            func.sum(TopSKU.quantity).label('total_qty')
        ).filter(
            TopSKU.analysis_id.in_(analysis_ids)
        ).group_by(TopSKU.sku).order_by(
            func.sum(TopSKU.quantity).desc()
        ).limit(5).all()

        context['chart_sku_labels'] = [item.sku for item in sku_data]
        context['chart_sku_data'] = [int(item.total_qty) for item in sku_data]

        # --- 5c. AGGREGATE TOP STATES ---
        state_data = db.session.query(
            StateSales.state,
            func.sum(StateSales.quantity).label('total_qty')
        ).filter(
            StateSales.analysis_id.in_(analysis_ids)
        ).group_by(StateSales.state).order_by(
            func.sum(StateSales.quantity).desc()
        ).limit(5).all()

        context['chart_state_labels'] = [item.state for item in state_data]
        context['chart_state_data'] = [int(item.total_qty) for item in state_data]

        # --- 5d. COST BREAKDOWN (Product Cost + User Expenses) ---
        # Product cost from transactions
        product_cost = metrics_sum.get('total_cost', 0.0)

        
        logger.info(f"💰 Product Cost calculated: ₹{product_cost:,.2f}")

        # Get all expense metrics (user-added expenses like Marketing, Packaging, etc.)
        expense_labels = []
        expense_values = []
        
        for metric_name, value in metrics_sum.items():
            # Filter only user-added expenses (they start with 'expense_')
            if metric_name.startswith('expense_'):
                # Convert 'expense_marketing' to 'Marketing'
                display_name = metric_name.replace('expense_', '').replace('_', ' ').title()
                expense_labels.append(display_name)
                expense_values.append(round(float(value), 2))
                logger.info(f"💸 Expense: {display_name} = ₹{value:,.2f}")

        # Build cost chart: Product Cost first, then all user expenses
        context['chart_cost_labels'] = ['Product Cost'] + expense_labels
        context['chart_cost_data'] = [round(product_cost, 2)] + expense_values
        
        logger.info(f"📊 Cost Chart Labels: {context['chart_cost_labels']}")
        logger.info(f"📊 Cost Chart Data: {context['chart_cost_data']}")

        # --- 5e. TREND CHART (Daily Revenue & Profit) ---
        transactions = Transaction.query.filter(
            Transaction.analysis_id.in_(analysis_ids)
        ).order_by(Transaction.order_date).all()

        if transactions:
            df_trans = pd.DataFrame([{
                'date': t.order_date,
                'revenue': t.total_amount or 0,
                'cost': (t.total_cost or 0) + (t.amz_fees or 0)
            } for t in transactions])

            if not df_trans.empty:
                daily_data = df_trans.groupby('date').sum().reset_index()
                daily_data['profit'] = daily_data['revenue'] - daily_data['cost']

                context['chart_trend_labels'] = daily_data['date'].astype(str).tolist()
                context['chart_trend_revenue'] = daily_data['revenue'].round(2).tolist()
                context['chart_trend_profit'] = daily_data['profit'].round(2).tolist()

    except Exception as e:
        logger.error(f"Error loading dashboard data: {e}", exc_info=True)

    return render_template('index.html', **context)


@app.route('/analyzer')
def analyzer():
    """Serves the profit analyzer page and RELOADS previous data."""
    context = {
        'data_exists': False,
        'summary': None,
        'expenses': {},
        'top_10_skus': [],
        'top_10_returns': [],
        'top_states': [],      # Added default
        'unpaid_orders': [],   # Added default
        'sku_health': []       # Added default
    }

    output_filename = session.get('output_filename')
    
    if output_filename:
        loaded_data = load_analysis_from_file(output_filename)
        
        if loaded_data:
            context['data_exists'] = True
            context['summary'] = loaded_data.get('summary_data')
            context['top_10_skus'] = loaded_data.get('top_10_skus', [])
            context['top_10_returns'] = loaded_data.get('top_10_returns', [])
            # 🔥 Pass the new data
            context['top_states'] = loaded_data.get('top_states', [])
            context['unpaid_orders'] = loaded_data.get('unpaid_orders', [])
            context['sku_health'] = loaded_data.get('sku_health', [])
            
            context['expenses'] = session.get('dynamic_expenses', {})
            logger.info(f"♻️ Reloading complete analysis for {output_filename}")

    return render_template('analyzer.html', **context)


@app.route('/content-generator')
def content_generator_page():
    """Serves the content generator page."""
    return render_template('content_generator.html')


@app.route('/marketlens')
def strategy_coach_page():
    """
    Renders the Market Research / Strategy Agent page.
    This agent uses external web tools (DuckDuckGo), so it does not 
    require the user to have uploaded a file previously.
    """
    return render_template('marketlens.html')


@app.route("/debug_session")
def debug_session():
    """Debug endpoint to check session contents"""
    output_filename = session.get('output_filename')
    
    debug_info = {
        "session_keys": list(session.keys()),
        "output_filename": output_filename,
        "file_exists": False,
        "session_permanent": session.permanent,
        "dynamic_expenses": session.get('dynamic_expenses', "Not Found")
    }
    
    if output_filename:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        debug_info["file_exists"] = os.path.exists(file_path)
        debug_info["file_path"] = file_path
    
    return jsonify(debug_info)

@app.route('/reset_analysis')
def reset_analysis():
    """Clears the session variables associated with analysis."""
    session.pop('output_filename', None)
    session.pop('dynamic_expenses', None)
    session.pop('analysis_timestamp', None)
    return jsonify({'success': True, 'message': 'Analysis cleared'})
# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Processes files and returns summary data.
    Saves ONLY the Master Report to the Output Folder.
    """
    temp_conversion_dir = None

    try:
        # 1. Create temp dir
        temp_conversion_dir = tempfile.mkdtemp()

        # Get inputs
        order_files = request.files.getlist('order_files')
        payment_files = request.files.getlist('payment_files')
        return_files = request.files.getlist('return_files')
        cost_price_file = request.files.get('cost_price_file')

        # Get Expenses
        dynamic_expenses = {}
        for key, value in request.form.items():
            if key.startswith("expense_"):
                expense_name = key.replace("expense_", "").replace("_", " ").title()
                try:
                    dynamic_expenses[expense_name] = float(value)
                except ValueError:
                    return jsonify({'success': False, 'error': f'Invalid amount for {expense_name}.'}), 400

        if not dynamic_expenses:
            return jsonify({'success': False, 'error': 'No expense inputs found.'}), 400

        if not order_files or not payment_files or not return_files or not cost_price_file:
            return jsonify({'success': False, 'error': 'Missing required files.'}), 400

        # Save Files
        order_paths, payment_paths, return_paths, cost_price_path = save_uploaded_files(
            order_files, payment_files, return_files, cost_price_file
        )

        try:
            # Conversion Logic
            converted_order_paths = []
            for file in order_paths:
                ext = os.path.splitext(file)[-1].lower()
                if ext in ['.txt', '.tsv']:
                    excel_path = convert_txt_to_excel(file, temp_conversion_dir)
                    converted_order_paths.append(excel_path)
                else:
                    converted_order_paths.append(file)

            converted_return_paths = []
            for file in return_paths:
                ext = os.path.splitext(file)[-1].lower()
                if ext in ['.txt', '.tsv']:
                    excel_path = convert_txt_to_excel(file, temp_conversion_dir)
                    converted_return_paths.append(excel_path)
                else:
                    converted_return_paths.append(file)

            start_date = request.form.get('start_date') or None
            end_date = request.form.get('end_date') or None

            # Process Data
            output_filename = process_data(
                order_files=converted_order_paths,
                payment_files=payment_paths,
                return_files=converted_return_paths,
                cost_price_file=cost_price_path,
                dynamic_expenses=dynamic_expenses,
                output_folder=app.config['OUTPUT_FOLDER'], 
                start_date=start_date,
                end_date=end_date
            )

            # Read Data
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Initialize containers
            summary_dict = {}
            top_10_list = []
            top_10_returns_list = []
            top_states_list = []
            unpaid_orders_list = []
            sku_health_rows = []
            merged_df = pd.DataFrame() # 🔥 FIXED: Initialize here to avoid crash

            # Load Sheets
            summary_df = pd.read_excel(output_path, sheet_name='Summary')
            top_10_df = pd.read_excel(output_path, sheet_name='Top 10 SKUs')
            top_10_list = top_10_df.to_dict('records')

            # Process Summary
            for _, row in summary_df.iterrows():
                metric = str(row['Metric']).strip()
                val = row['Value']
                try: val = float(val) 
                except: val = 0.0
                
                key_map = {'Total Quantity': 'total_quantity', 'Total Payment': 'total_payment', 
                           'Total Cost': 'total_cost', 'Net Profit': 'net_profit',
                           'Total Return Quantity': 'total_return_quantity', 
                           'Total Unpaid Orders': 'total_unpaid_orders',
                           'Total Amz Fees': 'total_amz_fees'}
                
                if metric in key_map: summary_dict[key_map[metric]] = val
                elif metric.lower() not in ['total quantity', 'total payment', 'total cost', 'net profit']:
                    summary_dict["expense_" + metric.lower().replace(" ", "_")] = val

            # Load secondary sheets
            try:
                top_10_returns_list = pd.read_excel(output_path, sheet_name='Top 10 Returns').to_dict('records')
            except: pass
            
            try:
                state_df = pd.read_excel(output_path, sheet_name='Top 10 States')
                state_df = state_df.rename(columns={'ship_state': 'state', 'quantity': 'total_orders'})
                top_states_list = state_df.to_dict('records')
            except: pass

            try:
                unpaid_orders_df = pd.read_excel(output_path, sheet_name='Unpaid Orders')
                unpaid_orders_list = unpaid_orders_df.to_dict('records')
            except: unpaid_orders_df = pd.DataFrame()

            # Re-run SKU Health
            try:
                merged_df = pd.read_excel(output_path, sheet_name='Merged Data')
                
                # 🔥 FIXED: Handle NaNs in merged_df for safer DB storage
                # This replaces NaN with None (which becomes NULL in SQL)
                merged_df = merged_df.where(pd.notnull(merged_df), None)

                if 'sku' in merged_df.columns:
                    health_res = analyze_sku_health(
                        merged_data=merged_df, 
                        returns=pd.DataFrame(top_10_returns_list), 
                        unpaid_orders=unpaid_orders_df, 
                        min_orders=1
                    )
                    s_df = health_res.scored_df
                    s_df['health_score_pred'] = s_df['health_score_pred'].round(2)
                    s_df['proxy_label'] = s_df['proxy_label'].round(2)
                    s_df['health_rating'] = s_df['health_score_pred'].apply(lambda x: "Excellent" if x>=75 else "Good" if x>=60 else "Fair" if x>=40 else "Poor")
                    s_df = s_df.rename(columns={'SKU':'SKU', 'health_score_pred':'Score', 'proxy_label':'Rating', 'health_rating':'Health', 'key_issues':'Key_Issues', 'anomaly':'Anomaly', 'cluster':'Segment'})
                    
                    disp = ['SKU', 'Score', 'Health', 'Key_Issues']
                    if 'Rating' in s_df.columns: disp.append('Rating')
                    if 'Anomaly' in s_df.columns: 
                        s_df['Anomaly'] = s_df['Anomaly'].map({True:'Yes', False:'No'})
                        disp.append('Anomaly')
                    
                    sku_health_rows = s_df[disp].to_dict('records')
            except Exception as e:
                logger.error(f"SKU Health/Merge error: {e}")

            # Save Session
            session['dynamic_expenses'] = dynamic_expenses 
            session['output_filename'] = output_filename
            session['analysis_timestamp'] = datetime.now().isoformat()
            session.modified = True
            
            # 🔥 FIXED: Safe conversion to dict
            merged_data_list = merged_df.to_dict('records') if not merged_df.empty else []

            # Save to DB
            save_full_analysis(
                file_name=output_filename, start_date=start_date, end_date=end_date,
                summary=summary_dict, top_skus=top_10_list, top_returns=top_10_returns_list,
                top_states=top_states_list, 
                merged_data=merged_data_list # Passing safe list
                # Removed: unpaid_orders, sku_health
            )

            # Response (Includes analysis data for UI, but it wasn't saved to DB)
            return jsonify({
                'success': True, 'filename': output_filename, 'summary': summary_dict,
                'top_10_skus': top_10_list, 'top_10_returns': top_10_returns_list,
                'top_states': top_states_list, 'unpaid_orders': unpaid_orders_list,
                'sku_health': sku_health_rows
            })

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': f'Processing error: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500
    
    finally:
        if temp_conversion_dir and os.path.exists(temp_conversion_dir):
            try:
                shutil.rmtree(temp_conversion_dir)
                logger.info(f"🧹 Cleaned up temp directory: {temp_conversion_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temp dir: {e}")

@app.route('/download/<filename>')
def download(filename):
    """Downloads the previously generated Excel file."""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Security check: ensure file exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Prevent directory traversal
        if not os.path.abspath(file_path).startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])):
            return jsonify({'error': 'Invalid file path'}), 403
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@app.route('/generate_content', methods=['POST'])
def generate_content_api():
    try:
        data = request.get_json()
        attributes = data.get("attributes", {})

        result = generate_product_content(attributes)

        # Ensure JSON-safe response
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/marketlens', methods=['POST'])
def unified_market_api():
    """
    Unified Endpoint: Handles Market Research, Product Scouting, and Chat.
    """
    try:
        data = request.get_json()
        # Accept 'niche' OR 'query' to be backward compatible
        user_query = data.get('niche') or data.get('query')

        if not user_query:
            return jsonify({'success': False, 'error': 'Please provide a query.'})

        logger.info(f"🤖 Unified Agent received: {user_query}")

        # Initialize Agent
        agent = UnifiedMarketAgent()

        # Run Async Logic
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(agent.process_request(user_query))

        # Result contains: {'type': '...', 'content': '...', 'products': [...]}
        return jsonify({
            'success': True,
            'report': result['content'], # Maps to 'content' in frontend
            'analysis': result['content'], # Backup key
            'products': result.get('products', []),
            'type': result.get('type', 'chat') # Tell frontend what type it was
        })

    except Exception as e:
        logger.error(f"Unified Agent Error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})
    
# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == '__main__':
    # 1. Create the tables if they don't exist
    with app.app_context():
        # This is CRITICAL. It looks at your imported models and creates the tables.
        database.create_all()
        print("✅ Database tables created (if they didn't exist).")

    # 2. Wipe old sessions (Your existing code)
    session_dir = app.config.get("SESSION_FILE_DIR", "./flask_session_data")
    
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir) 
            print(f"🧹 Cleaned up old session data in {session_dir}")
        except Exception as e:
            print(f"⚠️ Could not clean session data: {e}")
            
    os.makedirs(session_dir, exist_ok=True)

    # 3. Run the app
    app.run(debug=True)