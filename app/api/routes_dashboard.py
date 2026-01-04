import logging
from datetime import datetime

import pandas as pd
from flask import Blueprint, render_template, request
from sqlalchemy import func, extract

from app.db.connection import database as db
from app.db.models import Analysis, SummaryMetric, TopSKU, StateSales, Transaction

# Proper logging
logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    """
    Renders the Dashboard with optional month filtering.
    """

    # --- 1. GET AVAILABLE MONTHS ---
    available_months = []
    try:
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

    selected_month = request.args.get('month', None)

    # --- CONTEXT DEFAULTS ---
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

    try:
        # --- FILTER DATA ---
        if selected_month and selected_month != 'all':
            year, month = map(int, selected_month.split('-'))
            analyses = Analysis.query.filter(
                extract('year', Analysis.start_date) == year,
                extract('month', Analysis.start_date) == month
            ).all()
        else:
            analyses = Analysis.query.all()

        if not analyses:
            return render_template('index.html', **context)

        analysis_ids = [a.id for a in analyses]
        context['data_exists'] = True
        context['last_updated'] = analyses[-1].created_at.strftime('%B %d, %Y')

        # --- SUMMARY METRICS ---
        metrics_sum = {}
        for m in SummaryMetric.query.filter(
            SummaryMetric.analysis_id.in_(analysis_ids)
        ).all():
            metrics_sum[m.metric] = metrics_sum.get(m.metric, 0) + m.value

        revenue = metrics_sum.get('total_payment', 0)
        net_profit = metrics_sum.get('net_profit', 0)
        total_qty = metrics_sum.get('total_quantity', 0)
        return_qty = metrics_sum.get('total_return_quantity', 0)

        context['revenue'] = f"{revenue:,.2f}"
        context['net_profit'] = f"{net_profit:,.2f}"
        context['total_orders'] = f"{int(total_qty):,}"
        context['return_rate'] = f"{(return_qty / total_qty * 100) if total_qty else 0:.2f}"

        # --- TOP SKUs ---
        sku_data = db.session.query(
            TopSKU.sku,
            func.sum(TopSKU.quantity).label('total_qty')
        ).filter(
            TopSKU.analysis_id.in_(analysis_ids)
        ).group_by(TopSKU.sku).order_by(
            func.sum(TopSKU.quantity).desc()
        ).limit(5).all()

        context['chart_sku_labels'] = [s.sku for s in sku_data]
        context['chart_sku_data'] = [int(s.total_qty) for s in sku_data]

        # --- TOP STATES ---
        state_data = db.session.query(
            StateSales.state,
            func.sum(StateSales.quantity).label('total_qty')
        ).filter(
            StateSales.analysis_id.in_(analysis_ids)
        ).group_by(StateSales.state).order_by(
            func.sum(StateSales.quantity).desc()
        ).limit(5).all()

        context['chart_state_labels'] = [s.state for s in state_data]
        context['chart_state_data'] = [int(s.total_qty) for s in state_data]

        # --- COST BREAKDOWN ---
        product_cost = metrics_sum.get('total_cost', 0.0)

        expense_labels = []
        expense_values = []

        for name, value in metrics_sum.items():
            if name.startswith("expense_"):
                expense_labels.append(name.replace("expense_", "").replace("_", " ").title())
                expense_values.append(float(value))

        context['chart_cost_labels'] = ['Product Cost'] + expense_labels
        context['chart_cost_data'] = [round(product_cost, 2)] + expense_values

        # --- TREND CHART ---
        transactions = Transaction.query.filter(
            Transaction.analysis_id.in_(analysis_ids)
        ).order_by(Transaction.order_date).all()

        if transactions:
            df = pd.DataFrame([{
                'date': t.order_date,
                'revenue': t.total_amount or 0,
                'cost': (t.total_cost or 0) + (t.amz_fees or 0)
            } for t in transactions])

            if not df.empty:
                daily = df.groupby('date').sum().reset_index()
                daily['profit'] = daily['revenue'] - daily['cost']

                context['chart_trend_labels'] = daily['date'].astype(str).tolist()
                context['chart_trend_revenue'] = daily['revenue'].round(2).tolist()
                context['chart_trend_profit'] = daily['profit'].round(2).tolist()

    except Exception as e:
        logger.exception(f"Dashboard load failed: {e}")

    return render_template('index.html', **context)
