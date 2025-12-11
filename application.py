import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify, send_file, session
from werkzeug.utils import secure_filename
from content_generation import generate_product_content
from strategy_agent import generate_business_strategy
from config import Config
from ecommerce_analyzer import process_data
import pandas as pd
import chardet
import google.generativeai as genai
import json
import re
from dotenv import load_dotenv
from datetime import datetime
from sku_health import analyze_sku_health
import logging
from datetime import timedelta
import shutil

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

        # 4. 🔥 NEW: Read Top 10 States
        try:
            top_states_df = pd.read_excel(output_path, sheet_name='Top 10 States')
            # Rename columns to match frontend expectations
            if 'ship_state' in top_states_df.columns:
                top_states_df = top_states_df.rename(columns={'ship_state': 'state', 'quantity': 'total_orders'})
            top_states_list = top_states_df.to_dict('records')
        except Exception as e:
            logger.warning(f"Could not read Top 10 States: {e}")
            top_states_list = []

        # 5. 🔥 NEW: Read Unpaid Orders
        try:
            unpaid_orders_df = pd.read_excel(output_path, sheet_name='Unpaid Orders')
            unpaid_orders_list = unpaid_orders_df.to_dict('records')
        except Exception as e:
            logger.warning(f"Could not read Unpaid Orders: {e}")
            unpaid_orders_list = []
            unpaid_orders_df = pd.DataFrame() # Create empty for SKU health check

        # 6. 🔥 NEW: Re-Run SKU Health Analysis
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
    """Renders the main upload page."""
    return render_template('index.html')


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


@app.route('/strategy_agent')
def strategy_coach_page():
    """Strategy agent page - checks if we have a report file"""
    
    # Check if we have a saved filename in session
    output_filename = session.get('output_filename')
    data_exists = False
    
    if output_filename:
        # Verify file exists
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        data_exists = os.path.exists(file_path)
        logger.info(f"📊 Strategy page - File: {output_filename}, Exists: {data_exists}")
    else:
        logger.info("📊 Strategy page - No output filename in session")
    
    return render_template(
        "strategy_agent.html",
        data_exists=data_exists
    )


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
    Processes files and returns summary data as JSON.
    Stores ONLY the filename in session for later retrieval.
    """
    try:
        # Get File Lists from Form
        order_files = request.files.getlist('order_files')
        payment_files = request.files.getlist('payment_files')
        return_files = request.files.getlist('return_files')
        cost_price_file = request.files.get('cost_price_file')

        # Get Dynamic Expenses
        dynamic_expenses = {}
        for key, value in request.form.items():
            if key.startswith("expense_"):
                expense_name = key.replace("expense_", "").replace("_", " ").title()
                try:
                    expense_value = float(value)
                    dynamic_expenses[expense_name] = expense_value
                except ValueError:
                    return jsonify({'success': False, 'error': f'Invalid amount for {expense_name}.'}), 400

        if not dynamic_expenses:
            return jsonify({'success': False, 'error': 'No expense inputs found. Please provide at least one.'}), 400

        # Validation
        if not order_files or not payment_files or not return_files or not cost_price_file:
            return jsonify({'success': False, 'error': 'Missing one or more required file types.'}), 400

        # Save Files Securely
        order_paths, payment_paths, return_paths, cost_price_path = save_uploaded_files(
            order_files, payment_files, return_files, cost_price_file
        )

        # Run Processing Logic
        try:
            # Convert text files to Excel
            converted_order_paths = []
            for file in order_paths:
                ext = os.path.splitext(file)[-1].lower()
                if ext in ['.txt', '.tsv']:
                    excel_path = convert_txt_to_excel(file, app.config['OUTPUT_FOLDER'])
                    converted_order_paths.append(excel_path)
                else:
                    converted_order_paths.append(file)

            converted_return_paths = []
            for file in return_paths:
                ext = os.path.splitext(file)[-1].lower()
                if ext in ['.txt', '.tsv']:
                    excel_path = convert_txt_to_excel(file, app.config['OUTPUT_FOLDER'])
                    converted_return_paths.append(excel_path)
                else:
                    converted_return_paths.append(file)

            # Get date range
            start_date = request.form.get('start_date') or None
            end_date = request.form.get('end_date') or None

            # Process data
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

            # Read Data from Generated Excel
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            # Read Summary sheet
            summary_df = pd.read_excel(output_path, sheet_name='Summary')

            # Read Top 10 SKUs sheet
            top_10_df = pd.read_excel(output_path, sheet_name='Top 10 SKUs')
            
            # Read Unpaid Orders
            unpaid_orders_df = pd.read_excel(output_path, sheet_name='Unpaid Orders')
            unpaid_orders_list = unpaid_orders_df.to_dict('records')
            
            # Read Top 10 Returns
            try:
                top_10_returns_df = pd.read_excel(output_path, sheet_name='Top 10 Returns')
                top_10_returns_list = top_10_returns_df.to_dict('records')
            except Exception as e:
                logger.warning(f"Could not read Top 10 Returns sheet: {e}")
                top_10_returns_list = []

            # Read Top 10 States
            try:
                top_states_df = pd.read_excel(output_path, sheet_name='Top 10 States')
                top_states_df = top_states_df.rename(columns={
                    'ship_state': 'state',
                    'quantity': 'total_orders'
                })
                top_states_list = top_states_df.to_dict('records')
            except Exception as e:
                logger.warning(f"Could not read Top 10 States sheet: {e}")
                top_states_list = []

            # Run SKU Health Analysis
            sku_health_rows = []
            try:
                logger.info("Starting SKU health analysis...")
                
                # Load required data
                merged_data_df = pd.read_excel(output_path, sheet_name='Merged Data')
                
                # Check if we have SKU column
                if 'sku' not in merged_data_df.columns:
                    raise ValueError("Merged data must contain 'sku' column")
                
                # Load return data
                try:
                    return_df = pd.read_excel(output_path, sheet_name='Top 10 Returns')
                except Exception as e:
                    return_df = pd.DataFrame()
                
                # Run SKU health analysis
                health_result = analyze_sku_health(
                    merged_data=merged_data_df,
                    returns=return_df,
                    unpaid_orders=unpaid_orders_df,
                    min_orders=1
                )
                
                # Extract scored dataframe
                scored_df = health_result.scored_df
                
                # Round scores for display
                scored_df['health_score_pred'] = scored_df['health_score_pred'].round(2)
                scored_df['proxy_label'] = scored_df['proxy_label'].round(2)
                
                # Add health rating based on score
                def get_health_rating(score):
                    if score >= 75: return "Excellent"
                    elif score >= 60: return "Good"
                    elif score >= 40: return "Fair"
                    else: return "Poor"
                
                scored_df['health_rating'] = scored_df['health_score_pred'].apply(get_health_rating)
                
                # Rename columns for UI
                scored_df = scored_df.rename(columns={
                    'SKU': 'SKU',
                    'health_score_pred': 'Score',
                    'proxy_label': 'Rating',
                    'cluster': 'Segment',
                    'anomaly': 'Anomaly',
                    'health_rating': 'Health',
                    'key_issues': 'Key_Issues'
                })
                
                # Select columns for display
                display_cols = ['SKU', 'Score', 'Health', 'Key_Issues']
                if 'Rating' in scored_df.columns: display_cols.append('Rating')
                if 'Segment' in scored_df.columns: display_cols.append('Segment')
                if 'Anomaly' in scored_df.columns:
                    scored_df['Anomaly'] = scored_df['Anomaly'].map({True: 'Yes', False: 'No'})
                    display_cols.append('Anomaly')
                
                sku_health_rows = scored_df[display_cols].to_dict('records')
                
            except Exception as e:
                logger.error(f"SKU Health Score Model Failed: {e}", exc_info=True)
                sku_health_rows = []

            # Process Summary Data
            summary_dict = {}
            for _, row in summary_df.iterrows():
                metric = str(row['Metric']).strip()
                value = row['Value']

                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = 0.0

                key_mapping = {
                    'Total Quantity': 'total_quantity',
                    'Total Return Quantity': 'total_return_quantity',
                    'Total Unpaid Orders': 'total_unpaid_orders',
                    'Total Payment': 'total_payment',
                    'Total Cost': 'total_cost',
                    'Total Amz Fees': 'total_amz_fees',
                    'Net Profit': 'net_profit'
                }

                if metric in key_mapping:
                    summary_dict[key_mapping[metric]] = value
                elif metric.lower() not in ['total quantity', 'total payment', 'total cost', 'net profit']:
                    clean_key = "expense_" + metric.lower().replace(" ", "_")
                    summary_dict[clean_key] = value

            # Convert Top 10 SKUs to list of dictionaries
            top_10_list = top_10_df.to_dict('records')
            
            # 🔥🔥🔥 CRITICAL FIX: Save expenses AND filename to session 🔥🔥🔥
            session['dynamic_expenses'] = dynamic_expenses 
            session['output_filename'] = output_filename
            session['analysis_timestamp'] = datetime.now().isoformat()
            session.modified = True
            
            logger.info(f"✅ Session saved with filename: {output_filename} and expenses")
            
            # Return Complete Response
            return jsonify({
                'success': True,
                'filename': output_filename,
                'summary': summary_dict,
                'top_10_skus': top_10_list,
                'top_10_returns': top_10_returns_list,
                'top_states': top_states_list,
                'unpaid_orders': unpaid_orders_list,
                'sku_health': sku_health_rows
            })

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': f'Processing error: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


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

@app.route('/api/strategy-coach', methods=['POST'])
def strategy_coach():
    """
    Strategy coach endpoint - loads data from Excel file using stored filename.
    """
    try:
        # Get filename from session
        output_filename = session.get('output_filename')
        
        if not output_filename:
            logger.warning("⚠️ Strategy coach called but no filename in session")
            return jsonify({
                'success': False, 
                'error': 'No financial data found. Please run the Profit Analyzer first.'
            })
        
        # Load data from file
        logger.info(f"📂 Loading data from file: {output_filename}")
        data = load_analysis_from_file(output_filename)
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Failed to load analysis data. Please re-run the Profit Analyzer.'
            })
        
        summary_data = data.get('summary_data', {})
        top_10_returns = data.get('top_10_returns', [])
        
        logger.info("✅ Data loaded successfully from file")
        
        # Generate strategy report
        report_text = generate_business_strategy(summary_data, top_10_returns)
        
        return jsonify({
            'success': True, 
            'report': report_text
        })
        
    except Exception as e:
        logger.error(f"Strategy Coach Error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == '__main__':
    # 🔥 CRITICAL: Wipe old sessions when server starts
    # This ensures every time you restart the terminal, the site is empty.
    session_dir = app.config.get("SESSION_FILE_DIR", "./flask_session_data")
    
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir) # Delete the folder
            print(f"🧹 Cleaned up old session data in {session_dir}")
        except Exception as e:
            print(f"⚠️ Could not clean session data: {e}")
            
    # Recreate the empty folder
    os.makedirs(session_dir, exist_ok=True)

    app.run(debug=True)