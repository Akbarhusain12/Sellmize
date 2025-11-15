import os
from flask import (
    Flask, 
    render_template, 
    request, 
    redirect, 
    url_for, 
    send_from_directory, 
    flash,
    jsonify,
    send_file
)
from werkzeug.utils import secure_filename
from config import Config
from ecommerce_analyzer import process_data
import pandas as pd
import chardet
import google.generativeai as genai
import json, re
from dotenv import load_dotenv
from datetime import datetime
from sku_health import analyze_sku_health  # Updated import
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')


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
    logger.info(f"Detected encoding for {os.path.basename(txt_file_path)}: {detected_encoding} (confidence={confidence:.2f})")

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
                logger.info(f"Fallback to encoding: {fallback}")
                break
            except Exception:
                continue
        else:
            raise Exception(f"Could not read {txt_file_path} with any encoding: {e}")

    # --- Validate ---
    if df.empty:
        raise Exception(f"Conversion failed: {txt_file_path} produced empty DataFrame (encoding={detected_encoding}).")

    # --- Save as Excel ---
    base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    excel_path = os.path.join(output_folder, f"{base_name}.xlsx")
    df.to_excel(excel_path, index=False)
    logger.info(f"Converted {ext.upper()} → Excel: {excel_path} | {df.shape[0]} rows, {df.shape[1]} columns")

    return excel_path

      
@app.route('/analyzer')
def analyzer():
    """Serves the profit analyzer page."""
    return render_template('analyzer.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Processes files and returns summary data as JSON.
    """
    try:
        # --- 1. Get File Lists from Form ---
        order_files = request.files.getlist('order_files')
        payment_files = request.files.getlist('payment_files')
        return_files = request.files.getlist('return_files')
        cost_price_file = request.files.get('cost_price_file')

        # --- 2. Get Dynamic Expenses ---
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

        # --- 3. Validation ---
        if not order_files or not payment_files or not return_files or not cost_price_file:
            return jsonify({'success': False, 'error': 'Missing one or more required file types.'}), 400

        # --- 4. Save Files Securely ---
        order_paths, payment_paths, return_paths, cost_price_path = save_uploaded_files(
            order_files, payment_files, return_files, cost_price_file
        )

        # --- 5. Run Processing Logic ---
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
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            start_date = start_date if start_date else None
            end_date = end_date if end_date else None

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

            # --- 6. Read Data from Generated Excel ---
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

            # --- 7. Run SKU Health Analysis (UPDATED) ---
            sku_health_rows = []
            try:
                logger.info("Starting SKU health analysis...")
                
                # Load required data
                merged_data_df = pd.read_excel(output_path, sheet_name='Merged Data')
                logger.info(f"Loaded merged data: {merged_data_df.shape[0]} rows, {merged_data_df.shape[1]} columns")
                
                # Check if we have SKU column
                if 'sku' not in merged_data_df.columns:
                    logger.error("No 'sku' column found in merged data")
                    raise ValueError("Merged data must contain 'sku' column")
                
                # Log SKU distribution
                sku_counts = merged_data_df['sku'].value_counts()
                logger.info(f"Found {len(sku_counts)} unique SKUs")
                logger.info(f"Top 5 SKUs by row count:\n{sku_counts.head()}")
                
                # Load return data
                try:
                    return_df = pd.read_excel(output_path, sheet_name='Top 10 Returns')
                    logger.info(f"Loaded return data: {return_df.shape[0]} rows")
                except Exception as e:
                    logger.warning(f"Could not load return data: {e}")
                    return_df = pd.DataFrame()
                
                # Run improved SKU health analysis with minimum order filter = 1
                health_result = analyze_sku_health(
                    merged_data=merged_data_df,
                    returns=return_df,
                    unpaid_orders=unpaid_orders_df,
                    min_orders=1  # Changed to 1 to include all SKUs with at least one order
                )
                
                # Extract scored dataframe
                scored_df = health_result.scored_df
                logger.info(f"Health analysis produced {len(scored_df)} scored SKUs")
                
                # Round scores for display
                scored_df['health_score_pred'] = scored_df['health_score_pred'].round(2)
                scored_df['proxy_label'] = scored_df['proxy_label'].round(2)
                
                # Add health rating based on score
                def get_health_rating(score):
                    if score >= 75:
                        return "Excellent"
                    elif score >= 60:
                        return "Good"
                    elif score >= 40:
                        return "Fair"
                    else:
                        return "Poor"
                
                scored_df['health_rating'] = scored_df['health_score_pred'].apply(get_health_rating)
                
                # Rename columns for UI
                scored_df = scored_df.rename(columns={
                    'SKU': 'SKU',
                    'health_score_pred': 'Score',
                    'proxy_label': 'Rating',
                    'cluster': 'Segment',
                    'anomaly': 'Anomaly',
                    'health_rating': 'Health',
                    'key_issues': 'Key_Issues'  # Use underscore instead of space
                })
                
                # Select columns for display
                display_cols = ['SKU', 'Score', 'Health', 'Key_Issues']
                
                # Optionally include additional columns
                if 'Rating' in scored_df.columns:
                    display_cols.append('Rating')
                if 'Segment' in scored_df.columns:
                    display_cols.append('Segment')
                if 'Anomaly' in scored_df.columns:
                    # Convert boolean to Yes/No for better display
                    scored_df['Anomaly'] = scored_df['Anomaly'].map({True: 'Yes', False: 'No'})
                    display_cols.append('Anomaly')
                
                sku_health_rows = scored_df[display_cols].to_dict('records')
                
                logger.info(f"SKU health analysis complete: {len(sku_health_rows)} SKUs analyzed")
                logger.info(f"Model metrics: {health_result.train_metrics}")
                
                # Log sample results
                if len(sku_health_rows) > 0:
                    logger.info(f"Sample result: {sku_health_rows[0]}")
                
            except Exception as e:
                logger.error(f"SKU Health Score Model Failed: {e}", exc_info=True)
                sku_health_rows = []
                # Don't fail the entire request, just return empty health data

            # --- 8. Process Summary Data ---
            summary_dict = {}
            for _, row in summary_df.iterrows():
                metric = str(row['Metric']).strip()
                value = row['Value']

                # Convert safely to float
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = 0.0

                # Map known metrics
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
                    # Everything else is treated as dynamic expense
                    clean_key = "expense_" + metric.lower().replace(" ", "_")
                    summary_dict[clean_key] = value

            # Convert Top 10 SKUs to list of dictionaries
            top_10_list = top_10_df.to_dict('records')
            
            # --- 9. Return Complete Response ---
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
    """
    Downloads the previously generated Excel file.
    """
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Security check: ensure file exists and is in output folder
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Check if file is actually in output folder (prevent directory traversal)
        if not os.path.abspath(file_path).startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])):
            return jsonify({'error': 'Invalid file path'}), 403
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        return jsonify({'error': f'Download error: {str(e)}'}), 500


@app.route('/process', methods=['POST'])
def process_files():
    """
    Legacy endpoint for backward compatibility.
    Handles file uploads, triggers processing, and returns the file directly.
    """
    if request.method == 'POST':
        
        # --- 1. Get File Lists from Form ---
        order_files = request.files.getlist('order_files')
        payment_files = request.files.getlist('payment_files')
        return_files = request.files.getlist('return_files')
        cost_price_file = request.files.get('cost_price_file')

        # --- 2. Get Cost Inputs ---
        try:
            advertisement_cost = float(request.form.get('advertisement_cost', 0))
            stitching_cost = float(request.form.get('stitching_cost', 0))
        except ValueError:
            flash('Error: Costs must be valid numbers.')
            return redirect(url_for('index'))

        # --- 3. Validation ---
        if not order_files or not payment_files or not return_files or not cost_price_file:
            flash('Error: Missing one or more required file types.')
            return redirect(url_for('index'))

        # --- 4. Save Files Securely ---
        order_paths, payment_paths, return_paths, cost_price_path = save_uploaded_files(
            order_files, payment_files, return_files, cost_price_file
        )

        # --- 5. Run Processing Logic ---
        try:
            output_filename = process_data(
                order_files=order_paths,
                payment_files=payment_paths,
                return_files=return_paths,
                cost_price_file=cost_price_path,
                advertisement_cost=advertisement_cost,
                stitching_cost=stitching_cost,
                output_folder=app.config['OUTPUT_FOLDER']
            )

            # --- 6. Send File to User ---
            return send_from_directory(
                app.config['OUTPUT_FOLDER'], 
                output_filename, 
                as_attachment=True
            )

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            flash(f'An error occurred during processing: {str(e)}')
            return redirect(url_for('index'))


@app.route('/content-generator')
def content_generator_page():
    """Serves the content generator page."""
    return render_template('content_generator.html')


@app.route('/generate_content', methods=['POST'])
def generate_content():
    """
    AI-powered product content generator using Gemini.
    Accepts dynamic product attributes and returns title, description, and bullet points in JSON.
    """
    try:
        # --- 1. Parse JSON request ---
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON payload provided."}), 400

        product_attributes = data.get("attributes", {})
        if not product_attributes:
            return jsonify({"success": False, "error": "Missing 'attributes' field in request."}), 400

        # --- 2. Dynamically build product details text ---
        product_details = "\n".join([f"{k}: {v}" for k, v in product_attributes.items() if v])

        # --- 3. Construct prompt ---
        prompt = f"""
        You are a professional AI copywriter who writes e-commerce product content.

        Generate creative, persuasive, and SEO-friendly product content strictly in JSON format.

        ### Instructions:
        1. Write a short and catchy product title (with brand name and main feature).
        2. Write a two-paragraph description:
           - Paragraph 1: Introduce the product, highlight emotions and lifestyle appeal.
           - Paragraph 2: Explain features, benefits, and what makes it unique.
        3. Write 5 longer bullet points (1–2 sentences each) describing unique selling points.

        ### Product Details:
        {product_details}

        ### Output JSON Example:
        {{
          "title": "Generated product title",
          "description": "Two engaging paragraphs here...",
          "bullet_points": [
            "Detailed bullet point 1.",
            "Detailed bullet point 2.",
            "Detailed bullet point 3.",
            "Detailed bullet point 4.",
            "Detailed bullet point 5."
          ]
        }}

        Return only valid JSON — no extra text or explanations.
        """

        # --- 4. Generate AI response ---
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()

        # --- 5. Extract JSON from response ---
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return jsonify({
                "success": False,
                "error": "Gemini did not return valid JSON.",
                "raw_output": text
            }), 500

        result = json.loads(json_match.group())

        # --- 6. Return clean response ---
        return jsonify({
            "success": True,
            "content": result
        })

    except Exception as e:
        logger.error(f"Content generation error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)