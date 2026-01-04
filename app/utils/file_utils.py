import os
import pandas as pd
import chardet
from flask import current_app
from werkzeug.utils import secure_filename
import logging

logger = logging.getLogger(__name__)


def save_uploaded_files(order_files, payment_files, return_files, cost_price_file):
    """Helper function to save uploaded files and return their paths."""
    upload_dir = current_app.config['UPLOAD_FOLDER']
    
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