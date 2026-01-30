from flask import Blueprint, render_template, request, session, jsonify, send_file
import os
import tempfile
import shutil
from flask_login import current_user, login_required
import pandas as pd
from datetime import datetime
from flask import current_app as current

from app.utils.file_utils import save_uploaded_files, convert_txt_to_excel
from app.services.ecommerce_service import process_data
from app.services.analysis_service import load_analysis_from_file
from app.db.repositories import save_full_analysis   # adjust to your actual path

import logging

logger = logging.getLogger(__name__)

# page blueprint
analyzer_page = Blueprint("analyzer_page", __name__)

# api blueprint
analyzer_api = Blueprint("analyzer_api", __name__)


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
                output_folder=current.config['OUTPUT_FOLDER'], 
                start_date=start_date,
                end_date=end_date
            )

            # Read Data
            output_path = os.path.join(current.config['OUTPUT_FOLDER'], output_filename)
            
            # Initialize containers
            summary_dict = {}
            top_10_list = []
            top_10_returns_list = []
            top_states_list = []
            unpaid_orders_list = []
            merged_df = pd.DataFrame() 

            # Load Sheets
            summary_df = pd.read_excel(output_path, sheet_name='Summary')
            
            try:
                merged_df = pd.read_excel(output_path, sheet_name='Merged Data')
                merged_df.columns = (
                    merged_df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                    .str.replace("-", "_")
                )
                logger.info(f"✅ Loaded merged data: {len(merged_df)} rows")
            except Exception as e:
                logger.error(f"❌ Failed to load Merged Data: {e}")
                merged_df = pd.DataFrame()

            if 'real_revenue' not in merged_df.columns:
                logger.error("❌ real_revenue missing in merged data BEFORE DB save")
                return jsonify({
                    "success": False,
                    "error": "real_revenue missing in merged data. Analyzer output invalid."
                }), 500

            
            # --- FIX 1: Sanitize Top 10 SKUs ---
            top_10_df = pd.read_excel(output_path, sheet_name='Top 10 SKUs')
            top_10_list = top_10_df.where(pd.notnull(top_10_df), None).to_dict('records')

            # Process Summary
            for _, row in summary_df.iterrows():
                metric = str(row['Metric']).strip()
                val = row['Value']
                try: 
                    val = float(val) 
                    # --- FIX 2: Check for NaN in summary values ---
                    if pd.isna(val): val = 0.0
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
                # --- FIX 3: Sanitize Returns ---
                returns_df = pd.read_excel(output_path, sheet_name='Top 10 Returns')
                top_10_returns_list = returns_df.where(pd.notnull(returns_df), None).to_dict('records')
            except: pass
            
            try:
                # --- FIX 4: Sanitize Top States ---
                state_df = pd.read_excel(output_path, sheet_name='Top 10 States')
                state_df = state_df.rename(columns={'ship_state': 'state', 'quantity': 'total_orders'})
                top_states_list = state_df.where(pd.notnull(state_df), None).to_dict('records')
            except: pass

            try:
                # --- FIX 5: Sanitize Unpaid Orders (CRITICAL) ---
                unpaid_orders_df = pd.read_excel(output_path, sheet_name='Unpaid Orders')
                unpaid_orders_list = unpaid_orders_df.where(pd.notnull(unpaid_orders_df), None).to_dict('records')
            except: unpaid_orders_df = pd.DataFrame()


            # Save Session
            session['dynamic_expenses'] = dynamic_expenses 
            session['output_filename'] = output_filename
            session['analysis_timestamp'] = datetime.now().isoformat()
            session.modified = True
            
            merged_data_list = merged_df.to_dict('records') if not merged_df.empty else []

            # Save to DB
            save_full_analysis(
                user_id=current_user.id,
                file_name=output_filename, start_date=start_date, end_date=end_date,
                summary=summary_dict, top_skus=top_10_list, top_returns=top_10_returns_list,
                top_states=top_states_list, 
                merged_data=merged_data_list 
            )

            # Response
            return jsonify({
                'success': True, 'filename': output_filename, 'summary': summary_dict,
                'top_10_skus': top_10_list, 'top_10_returns': top_10_returns_list,
                'top_states': top_states_list, 'unpaid_orders': unpaid_orders_list
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