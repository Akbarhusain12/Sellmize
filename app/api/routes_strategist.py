from flask import Blueprint, render_template, session, flash, redirect, url_for, current_app
from flask_login import login_required
import os
import logging
from app.ml.strategist import generate_strategist_report

# Configure logging
logger = logging.getLogger(__name__)

# Define Blueprint
strategist_bp = Blueprint('strategist_bp', __name__)


@strategist_bp.route('/')
@login_required
def strategist_engine():
    """
    Renders the AI strategist Engine Page with forecasts and restock recommendations.
    Uses the SAME processed Excel file from the Profit Analyzer (no re-upload needed).
    Session persists during navigation but clears on page refresh.
    """
    try:
        # 1. Check Session for Filename (set by analyzer after processing)
        output_filename = session.get('output_filename')
        
        if not output_filename:
            return render_template(
                'strategist.html',
                success=False,
                reason="no_data"
            )


        # 2. Construct Full Path to Processed File
        file_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
        
        # 3. Validate File Exists
        if not os.path.exists(file_path):
            logger.warning(f"strategist Engine: File not found - {file_path}")
            flash("❌ Analysis file expired or not found. Please run analysis again.", "error")
            session.pop('output_filename', None)
            session.pop('analysis_timestamp', None)
            return redirect(url_for('analyzer_page.analyzer'))

        # 4. Generate strategist Report (Forecasting + Restock Logic)
        logger.info(f"Generating strategist report from: {output_filename}")
        strategist_data = generate_strategist_report(file_path)

        # 5. Handle Errors from ML Model
        if "error" in strategist_data:
            logger.error(f"strategist generation failed: {strategist_data['error']}")
            flash(f"🚨 Forecasting failed: {strategist_data['error']}", "error")
            
            # Provide helpful context about the error
            error_msg = strategist_data['error']
            if "No date column" in error_msg or "date" in error_msg.lower():
                flash("💡 Tip: Make sure your uploaded files contain proper date columns.", "info")
            elif "Not enough data points" in error_msg or "Insufficient" in error_msg:
                flash("💡 Tip: Upload at least 10 days of sales history for accurate forecasting.", "info")
            elif "column" in error_msg.lower():
                flash("💡 Tip: Check that your files have the standard Amazon report format.", "info")
            
            return redirect(url_for('analyzer_page.analyzer'))

        # 6. Add session metadata for the template
        strategist_data['analysis_timestamp'] = session.get('analysis_timestamp')
        strategist_data['output_filename'] = output_filename
        
        # 7. Render Template with Data
        logger.info("✅ strategist report generated successfully")
        return render_template('strategist.html', **strategist_data)
        
    except Exception as e:
        logger.exception("Unexpected error in strategist engine")
        flash(f"⚠️ An unexpected error occurred: {str(e)}", "error")
        return redirect(url_for('analyzer_page.analyzer'))


@strategist_bp.route('/refresh')
@login_required
def refresh_strategist():
    """
    Force regenerate strategist report with fresh predictions.
    Useful if user wants to see updated forecasts without re-uploading files.
    """
    try:
        output_filename = session.get('output_filename')
        
        if not output_filename:
            return render_template(
                'strategist.html',
                success=False,
                reason="no_data"
            )

            
        file_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
        
        if not os.path.exists(file_path):
            flash("❌ Source file not found. Please re-run analysis.", "error")
            session.pop('output_filename', None)
            return redirect(url_for('analyzer_page.analyzer'))
        
        # Regenerate
        logger.info(f"Refreshing strategist report: {output_filename}")
        strategist_data = generate_strategist_report(file_path)
        
        if "error" in strategist_data:
            flash(f"🚨 Refresh failed: {strategist_data['error']}", "error")
            return redirect(url_for('strategist_bp.strategist_engine'))
        
        # Add metadata
        strategist_data['analysis_timestamp'] = session.get('analysis_timestamp')
        strategist_data['output_filename'] = output_filename
        
        flash("✅ strategist report refreshed successfully!", "success")
        return render_template('strategist.html', **strategist_data)
        
    except Exception as e:
        logger.exception("Error refreshing strategist")
        flash(f"⚠️ Refresh error: {str(e)}", "error")
        return redirect(url_for('strategist_bp.strategist_engine'))


@strategist_bp.route('/debug')
@login_required
def debug_strategist():
    """
    Debug endpoint to check if strategist engine can access analyzer data.
    Access via: /strategist/debug
    """
    output_filename = session.get('output_filename')
    
    debug_info = {
        "session_output_filename": output_filename,
        "file_exists": False,
        "file_path": None,
        "session_timestamp": session.get('analysis_timestamp'),
        "output_folder_exists": os.path.exists(current_app.config['OUTPUT_FOLDER']),
        "output_folder_path": current_app.config['OUTPUT_FOLDER'],
        "session_keys": list(session.keys()),
        "session_permanent": session.permanent
    }
    
    if output_filename:
        file_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
        debug_info["file_exists"] = os.path.exists(file_path)
        debug_info["file_path"] = file_path
        
        if os.path.exists(file_path):
            # Check what sheets are available
            try:
                import pandas as pd
                xl = pd.ExcelFile(file_path)
                debug_info["available_sheets"] = xl.sheet_names
                
                # Check if Merged Data sheet has required columns
                df = pd.read_excel(file_path, sheet_name='Merged Data', nrows=5)
                debug_info["merged_data_columns"] = df.columns.tolist()
                debug_info["merged_data_rows"] = len(pd.read_excel(file_path, sheet_name='Merged Data'))
            except Exception as e:
                debug_info["file_read_error"] = str(e)
    
    from flask import jsonify
    return jsonify(debug_info)