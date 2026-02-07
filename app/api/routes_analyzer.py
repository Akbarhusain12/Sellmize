from flask import Blueprint, render_template, request, session, jsonify, send_file
import os
import tempfile
import shutil
import math  # Needed for NaN checking
from flask_login import current_user, login_required
import pandas as pd
from datetime import datetime
from flask import current_app as current

from app.utils.file_utils import save_uploaded_files, convert_txt_to_excel
from app.services.ecommerce_service import process_data
from app.services.analysis_service import load_analysis_from_file
from app.db.repositories import save_full_analysis

import logging

logger = logging.getLogger(__name__)

# page blueprint
analyzer_page = Blueprint("analyzer_page", __name__)

# api blueprint
analyzer_api = Blueprint("analyzer_api", __name__)

# ----------------------- HELPER -----------------------

def clean_for_json(obj):
    """
    Recursively replaces NaN, Infinity, and -Infinity with None.
    This prevents the 'Invalid JSON' error in the frontend console.
    """
    if isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

# ----------------------- UI PAGE -----------------------

@analyzer_page.route('/')
@login_required
def analyzer():
    """Serves the profit analyzer page and reloads previous data."""
    context = {
        'data_exists': False,
        'summary': None,
        'expenses': {},
        'top_10_skus': [],
        'top_10_returns': [],
        'top_states': [],
        'unpaid_orders': []
    }

    output_filename = session.get('output_filename')

    if output_filename:
        loaded_data = load_analysis_from_file(output_filename)
        if loaded_data:
            context['data_exists'] = True
            context['summary'] = loaded_data.get('summary_data')
            context['top_10_skus'] = loaded_data.get('top_10_skus', [])
            context['top_10_returns'] = loaded_data.get('top_10_returns', [])
            context['top_states'] = loaded_data.get('top_states', [])
            context['unpaid_orders'] = loaded_data.get('unpaid_orders', [])
            context['expenses'] = session.get('dynamic_expenses', {})

    return render_template('analyzer.html', **context)


@analyzer_api.route('/analyze', methods=['POST'])
def analyze():
    temp_conversion_dir = None
    try:
        temp_conversion_dir = tempfile.mkdtemp()

        # Inputs
        order_files = request.files.getlist('order_files')
        payment_files = request.files.getlist('payment_files')
        return_files = request.files.getlist('return_files')
        cost_price_file = request.files.get('cost_price_file')

        # Expenses
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
                if os.path.splitext(file)[-1].lower() in ['.txt', '.tsv']:
                    converted_order_paths.append(convert_txt_to_excel(file, temp_conversion_dir))
                else:
                    converted_order_paths.append(file)

            converted_return_paths = []
            for file in return_paths:
                if os.path.splitext(file)[-1].lower() in ['.txt', '.tsv']:
                    converted_return_paths.append(convert_txt_to_excel(file, temp_conversion_dir))
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
                output_folder=current.config['OUTPUT_FOLDER'], 
                start_date=start_date,
                end_date=end_date
            )

            output_path = os.path.join(current.config['OUTPUT_FOLDER'], output_filename)
            summary_dict = {}
            
            # Load Sheets
            summary_df = pd.read_excel(output_path, sheet_name='Summary')
            
            try:
                merged_df = pd.read_excel(output_path, sheet_name='Merged Data')
                merged_df.columns = merged_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
            except Exception as e:
                logger.error(f"❌ Failed to load Merged Data: {e}")
                merged_df = pd.DataFrame()

            # ===== UPDATED: Check for 'revenue' column instead of 'real_revenue' =====
            if 'revenue' not in merged_df.columns:
                logger.warning("⚠️ 'revenue' column not found in Merged Data")

            # Sanitize Data Frames (where.pd.notnull converts NaN to None/null)
            top_10_df = pd.read_excel(output_path, sheet_name='Top 10 SKUs')
            top_10_list = top_10_df.where(pd.notnull(top_10_df), None).to_dict('records')

            # ===== UPDATED: New metric mapping for revenue-based calculations =====
            for _, row in summary_df.iterrows():
                metric = str(row['Metric']).strip()
                val = row['Value']
                try: 
                    val = float(val) 
                    if pd.isna(val): 
                        val = 0.0
                except: 
                    val = 0.0
                
                # Updated key mapping for new metrics
                key_map = {
                    'Total Quantity Ordered': 'total_quantity',
                    'Total Revenue (All Orders)': 'total_revenue',
                    'Total Payment Received': 'total_payment_received',
                    'Total Pending Payment': 'total_pending_payment',
                    'Total Unpaid Order Quantity': 'total_unpaid_quantity',
                    'Total Cost': 'total_cost',
                    'Total Amz Fees': 'total_amz_fees',
                    'Total Return Quantity': 'total_return_quantity',
                    'Total Expenses': 'total_expenses',
                    'Net Profit': 'net_profit',
                    
                    # Legacy support (old metric names)
                    'Total Quantity': 'total_quantity',
                    'Total Payment': 'total_payment_received',
                    'Total Unpaid Orders': 'total_unpaid_quantity',
                }
                
                if metric in key_map: 
                    summary_dict[key_map[metric]] = val
                else:
                    # Store as dynamic expense
                    clean_key = metric.lower().replace(" ", "_").replace("-", "_")
                    if clean_key not in ['total_quantity', 'total_payment', 'total_cost', 
                                         'net_profit', 'total_return_quantity', 
                                         'total_unpaid_orders', 'total_amz_fees']:
                        summary_dict[f"expense_{clean_key}"] = val

            # Load secondary sheets safely
            try:
                r_df = pd.read_excel(output_path, sheet_name='Top 10 Returns')
                top_10_returns_list = r_df.where(pd.notnull(r_df), None).to_dict('records')
            except Exception as e:
                logger.warning(f"Could not load Top 10 Returns: {e}")
                top_10_returns_list = []
            
            try:
                s_df = pd.read_excel(output_path, sheet_name='Top 10 States')
                # Handle both old and new column names
                if 'ship_state' in s_df.columns:
                    s_df = s_df.rename(columns={'ship_state': 'state', 'quantity': 'total_orders'})
                top_states_list = s_df.where(pd.notnull(s_df), None).to_dict('records')
            except Exception as e:
                logger.warning(f"Could not load Top 10 States: {e}")
                top_states_list = []

            try:
                u_df = pd.read_excel(output_path, sheet_name='Unpaid Orders')
                unpaid_orders_list = u_df.where(pd.notnull(u_df), None).to_dict('records')
            except Exception as e:
                logger.warning(f"Could not load Unpaid Orders: {e}")
                unpaid_orders_list = []

            # ===== UPDATED: Add default values for new metrics =====
            default_summary_keys = {
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
            
            for key, default_val in default_summary_keys.items():
                if key not in summary_dict:
                    summary_dict[key] = default_val

            # Session Management
            session['dynamic_expenses'] = dynamic_expenses 
            session['output_filename'] = output_filename
            session.modified = True
            
            # DB Persistence
            merged_data_list = merged_df.to_dict('records') if not merged_df.empty else []
            save_full_analysis(
                user_id=current_user.id,
                file_name=output_filename, 
                start_date=start_date, 
                end_date=end_date,
                summary=summary_dict, 
                top_skus=top_10_list, 
                top_returns=top_10_returns_list,
                top_states=top_states_list, 
                merged_data=merged_data_list 
            )

            # FINAL STEP: Wrap in clean_for_json to prevent Heroku 'Invalid JSON' crash
            response_payload = {
                'success': True, 
                'filename': output_filename, 
                'summary': summary_dict,
                'top_10_skus': top_10_list, 
                'top_10_returns': top_10_returns_list,
                'top_states': top_states_list, 
                'unpaid_orders': unpaid_orders_list
            }

            logger.warning(f"✅ Analysis complete. Sending response with {len(unpaid_orders_list)} unpaid orders.")
            
            return jsonify(clean_for_json(response_payload))

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
    
    finally:
        if temp_conversion_dir and os.path.exists(temp_conversion_dir):
            shutil.rmtree(temp_conversion_dir)

@analyzer_api.route('/download/<filename>')
def download(filename):
    """Downloads the previously generated Excel file."""
    try:
        file_path = os.path.join(current.config['OUTPUT_FOLDER'], filename)
        
        # Security check: ensure file exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Prevent directory traversal
        if not os.path.abspath(file_path).startswith(os.path.abspath(current.config['OUTPUT_FOLDER'])):
            return jsonify({'error': 'Invalid file path'}), 403
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@analyzer_api.route('/reset_analysis')
def reset_analysis():
    """Clears the session variables associated with analysis."""
    session.pop('output_filename', None)
    session.pop('dynamic_expenses', None)
    session.pop('analysis_timestamp', None)
    return jsonify({'success': True, 'message': 'Analysis cleared'})

@analyzer_api.route("/debug_session")
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
        file_path = os.path.join(current.config['OUTPUT_FOLDER'], output_filename)
        debug_info["file_exists"] = os.path.exists(file_path)
        debug_info["file_path"] = file_path
    
    return jsonify(debug_info)