import numpy as np
import pandas as pd
from prophet import Prophet
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def generate_strategist_report(file_path):
    """
    Enhanced Strategist Engine matching the working Jupyter notebook logic.
    Generates forecasts for 7, 15, and 30 days with accurate revenue predictions.
    """
    try:
        # ==========================================
        # STEP 1: LOAD & VALIDATE DATA
        # ==========================================
        logger.info(f"Loading data from: {file_path}")
        
        # Load the Merged Data sheet (contains all processed data)
        df = pd.read_excel(file_path, sheet_name='Merged Data')
        
        # Normalize column names (critical for consistency)
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('-', '_')
        )
        
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Find date column (flexible matching)
        date_column = None
        for col in ['order_date', 'purchase_date', 'date', 'ds']:
            if col in df.columns:
                date_column = col
                break
        
        if not date_column:
            # Try partial match
            date_cols = [c for c in df.columns if 'date' in c and 'purchase' in c]
            if date_cols:
                date_column = date_cols[0]
            else:
                return {"error": "No date column found. Required: 'order_date', 'purchase_date', or similar."}

        orders_weekday_df = pd.read_excel(file_path, sheet_name='Orders_Weekday')
        daily_orders_df = pd.read_excel(file_path, sheet_name='Daily_Orders')
        daily_revenue_df = pd.read_excel(file_path, sheet_name='Daily_Revenue')
        return_reasons_df = pd.read_excel(file_path, sheet_name='Return_Reasons')

        # ==========================================
        # STEP 2: DATA CLEANING (MATCHING JUPYTER LOGIC)
        # ==========================================
        
        # Convert to datetime (handle timezone issues)
        df['ds'] = pd.to_datetime(df[date_column], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['ds'])
        df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone
        
        if len(df) < 10:
            return {"error": f"Insufficient data. Found {len(df)} records, need at least 10 for forecasting."}

        # Filter cancelled orders (matching your Jupyter logic)
        if 'order_status' in df.columns:
            df = df[df['order_status'] != 'Cancelled']
        elif 'status' in df.columns:
            df = df[~df['status'].astype(str).str.contains('Cancel', case=False, na=False)]

        # Clean numeric columns
        numeric_cols = ['total', 'quantity', 'total_cost', 'item_price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # ==========================================
        # STEP 3: SMART PRICE RECOVERY (JUPYTER LOGIC)
        # ==========================================
        
        # Find price and SKU columns
        price_col = None
        sku_col = None
        
        for col in df.columns:
            if 'price' in col and 'item' in col:
                price_col = col
            if 'sku' in col and 'merchant' not in col:
                sku_col = col
        
        if price_col and sku_col:
            logger.info(f"Using price column: {price_col}, SKU column: {sku_col}")
            
            # Calculate median price per SKU (excluding 0s)
            price_map = df[df[price_col] > 0].groupby(sku_col)[price_col].median()
            
            def fill_price(row):
                """Fill missing/zero prices with SKU median"""
                if pd.isna(row[price_col]) or row[price_col] == 0:
                    return price_map.get(row[sku_col], 0)
                return row[price_col]
            
            df['cleaned_price'] = df.apply(fill_price, axis=1)
            
            # Recalculate revenue with cleaned prices
            if 'real_revenue' not in df.columns:
                df['real_revenue'] = df['quantity'] * df['cleaned_price']
            revenue_col = 'real_revenue'
            
            logger.info(f"Price recovery complete. Recovered {(df['cleaned_price'] > 0).sum()} prices.")
        else:
            # Fallback to 'total' column
            revenue_col = 'total'
            logger.warning("Could not find price/SKU columns. Using 'total' for revenue.")

        # ==========================================
        # STEP 4: REVENUE FORECAST (CLIFF REMOVAL)
        # ==========================================
        
        # Aggregate daily revenue
        daily_rev = df.groupby('ds')[revenue_col].sum().reset_index()
        daily_rev.columns = ['ds', 'y']
        daily_rev = daily_rev.sort_values('ds')  # Ensure chronological order
        
        # CRITICAL FIX: Remove last 3 days for TRAINING (cliff effect prevention)
        # But keep the ACTUAL last date for prediction window
        actual_last_date = daily_rev['ds'].max()  # Real last date in data
        daily_rev_clean = daily_rev.iloc[:-3].copy()  # Training data (trimmed)
        training_last_date = daily_rev_clean['ds'].max()
        
        logger.info(f"Actual last date in data: {actual_last_date}")
        logger.info(f"Training data ends on: {training_last_date}")
        logger.info(f"Data points for training: {len(daily_rev_clean)}")

        # Disable Prophet logging (suppress warnings)
        logging.getLogger('cmdstanpy').disabled = True
        logging.getLogger('prophet').disabled = True
        
        # Train Prophet Model (matching Jupyter parameters)
        m_rev = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,  # Disable for short datasets
            changepoint_prior_scale=0.05,  # Less aggressive
            seasonality_mode='additive'
        )
        m_rev.add_country_holidays(country_name='IN')
        if len(daily_rev_clean) < 14:
            return {"error": "Not enough historical data for forecasting (need ≥14 days)."}
        m_rev.fit(daily_rev_clean)
        
        # Predict 30 days into the future
        future = m_rev.make_future_dataframe(periods=30)
        forecast = m_rev.predict(future)

        # ==========================================
        # STEP 5: GENERATE MULTI-TIMEFRAME FORECASTS
        # ==========================================
        forecasts = {}
        
        for days in [7, 15, 30]:
            # Get predictions AFTER actual last data date (includes the 3 trimmed days in forecast)
            period_forecast = forecast[forecast['ds'] > actual_last_date].head(days)
            
            forecasts[f'{days}day'] = {
                'labels': period_forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'data': period_forecast['yhat'].clip(lower=0).round(2).tolist(),
                'lower': period_forecast['yhat_lower'].clip(lower=0).round(2).tolist(),
                'upper': period_forecast['yhat_upper'].clip(lower=0).round(2).tolist(),
                'total_revenue': round(period_forecast['yhat'].sum(), 2),
                'avg_daily_revenue': round(period_forecast['yhat'].mean(), 2),
                'min_revenue': round(period_forecast['yhat_lower'].sum(), 2),
                'max_revenue': round(period_forecast['yhat_upper'].sum(), 2)
            }

        # ==========================================
        # STEP 6: WEEKLY BREAKDOWN FOR 30-DAY VIEW
        # ==========================================
        forecast_30d = forecast[forecast['ds'] > actual_last_date].head(30).copy()
        forecast_30d['week'] = ((forecast_30d['ds'] - forecast_30d['ds'].min()).dt.days // 7) + 1
        
        weekly_forecast = forecast_30d.groupby('week').agg({
            'yhat': 'sum',
            'ds': 'min'
        }).reset_index()
        
        weekly_data = {
            'labels': [f"Week {int(w)}" for w in weekly_forecast['week']],
            'data': weekly_forecast['yhat'].round(2).tolist(),
            'dates': [d.strftime('%b %d') for d in weekly_forecast['ds']],
            'total': round(weekly_forecast['yhat'].sum(), 2)
        }

        # ==========================================
        # STEP 7: TOP SKU RESTOCK PLAN
        # ==========================================
        
        if not sku_col:
            # Try to find SKU column
            sku_candidates = [c for c in df.columns if 'sku' in c and 'merchant' not in c]
            sku_col = sku_candidates[0] if sku_candidates else None
        
        restock_data = []
        
        if sku_col:
            # Get top 10 SKUs by total quantity sold
            top_skus = (
                df.groupby(sku_col)['quantity']
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .index.tolist()
            )
            
            logger.info(f"Processing {len(top_skus)} SKUs for restock plan")

            for sku in top_skus:
                try:
                    sku_df = df[df[sku_col] == sku].copy()
                    
                    # Aggregate daily units
                    daily_units = sku_df.groupby('ds')['quantity'].sum().reset_index()
                    daily_units.columns = ['ds', 'y']

                    if len(daily_units) < 5:
                        continue

                    # Train SKU-specific model
                    m_sku = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=False
                    )
                    m_sku.add_country_holidays(country_name='IN')
                    m_sku.fit(daily_units)
                    
                    future_sku = m_sku.make_future_dataframe(periods=30)
                    fc_sku = m_sku.predict(future_sku)
                    
                    # Get last date for this SKU
                    last_sku_date = daily_units['ds'].max()
                    
                    # Predictions AFTER last data (matching Jupyter)
                    future_only = fc_sku[fc_sku['ds'] > last_sku_date].head(30)
                    expected_sales = max(0, future_only['yhat'].sum())
                    
                    # Safety stock = 20% buffer (matching Jupyter logic)
                    safety_stock = expected_sales * 0.20
                    
                    # Calculate Financials
                    total_sales = sku_df[revenue_col].sum() if revenue_col in sku_df.columns else 0
                    total_cost = sku_df['total_cost'].sum() if 'total_cost' in sku_df.columns else 0
                    net_margin = ((total_sales - total_cost) / total_sales * 100) if total_sales > 0 else 0
                    
                    # Return Rate
                    total_units = sku_df['quantity'].sum()
                    return_units = 0
                    
                    # Check for return indicators
                    if 'status' in sku_df.columns:
                        return_units = sku_df[
                            sku_df['status'].astype(str).str.contains('Return', case=False, na=False)
                        ]['quantity'].sum()
                    
                    return_rate = (return_units / total_units * 100) if total_units > 0 else 0

                    # Velocity Score (units sold per day)
                    days_active = (sku_df['ds'].max() - sku_df['ds'].min()).days + 1
                    velocity = total_units / max(1, days_active)

                    # Verdict Logic (matching Jupyter)
                    verdict = "✅ RESTOCK"
                    verdict_class = "text-green-600 bg-green-50 border-green-200"
                    priority = "High"
                    
                    if return_rate > 15:
                        verdict = "❌ KILL (High Returns)"
                        verdict_class = "text-red-600 bg-red-50 border-red-200"
                        priority = "Do Not Buy"
                    elif net_margin < 10 and expected_sales < 500:
                        verdict = "⚠️ REVIEW (Low Margin)"
                        verdict_class = "text-orange-600 bg-orange-50 border-orange-200"
                        priority = "Medium"
                    elif velocity < 1:
                        verdict = "⏸️ SLOW MOVER"
                        verdict_class = "text-blue-600 bg-blue-50 border-blue-200"
                        priority = "Low"

                    restock_data.append({
                        'sku': sku,
                        'predicted_sales': int(expected_sales),
                        'safety_stock': int(safety_stock),
                        'recommended_order': int(expected_sales + safety_stock),
                        'margin': round(net_margin, 1),
                        'return_rate': round(return_rate, 1),
                        'velocity': round(velocity, 1),
                        'priority': priority,
                        'verdict': verdict,
                        'verdict_class': verdict_class
                    })
                except Exception as e:
                    logger.warning(f"SKU {sku} forecast failed: {e}")
                    continue

            # Sort by priority
            priority_order = {'High': 1, 'Medium': 2, 'Low': 3, 'Do Not Buy': 4}
            restock_data.sort(key=lambda x: (priority_order.get(x['priority'], 5), -x['predicted_sales']))

        # ==========================================
        # STEP 8: GENERATE CHART DATA FOR FRONTEND
        # ==========================================
        

        weekday_chart = {
            'labels': orders_weekday_df['weekday'].tolist(),
            'data': orders_weekday_df['order_count'].tolist()
        }

        daily_orders_chart = {
            'labels': pd.to_datetime(daily_orders_df['order_date']).dt.strftime('%b %d').tolist(),
            'data': daily_orders_df['order_count'].tolist()
        }

        daily_revenue_chart = {
            'labels': pd.to_datetime(daily_revenue_df['order_date']).dt.strftime('%b %d').tolist(),
            'data': daily_revenue_df['revenue'].round(2).tolist()
        }

        if not return_reasons_df.empty:
            return_reasons_chart = {
                'labels': return_reasons_df['reason'].tolist(),
                'data': return_reasons_df['count'].tolist()
            }
        else:
            return_reasons_chart = {
                'labels': ['No Returns'],
                'data': [0]
            }

        # ==========================================
        # STEP 9: BUSINESS INSIGHTS & ALERTS
        # ==========================================
        insights = []
        
        # Revenue Trend Analysis
        historical_30d_revenue = daily_rev_clean['y'].tail(30).sum()
        predicted_30d_revenue = forecasts['30day']['total_revenue']
        
        if historical_30d_revenue > 0:
            revenue_change = ((predicted_30d_revenue - historical_30d_revenue) / historical_30d_revenue * 100)
        else:
            revenue_change = 0
        
        if revenue_change > 10:
            insights.append({
                'type': 'success',
                'icon': '📈',
                'title': 'Growth Opportunity',
                'message': f'Revenue projected to grow {revenue_change:.1f}% over next 30 days!'
            })
        elif revenue_change < -10:
            insights.append({
                'type': 'warning',
                'icon': '📉',
                'title': 'Revenue Alert',
                'message': f'Revenue may decline by {abs(revenue_change):.1f}%. Consider promotions or marketing push.'
            })
        else:
            insights.append({
                'type': 'info',
                'icon': '➡️',
                'title': 'Stable Revenue',
                'message': f'Revenue expected to remain stable (±{abs(revenue_change):.1f}%).'
            })
        
        # High Return Products
        high_return_count = len([x for x in restock_data if x['return_rate'] > 15])
        if high_return_count > 0:
            insights.append({
                'type': 'danger',
                'icon': '⚠️',
                'title': 'Quality Alert',
                'message': f'{high_return_count} products have >15% return rate. Review quality and descriptions.'
            })
        
        # Low Margin Warning
        low_margin_count = len([x for x in restock_data if x['margin'] < 10])
        if low_margin_count > 0:
            insights.append({
                'type': 'warning',
                'icon': '💰',
                'title': 'Margin Alert',
                'message': f'{low_margin_count} products have <10% margin. Optimize pricing or reduce costs.'
            })

        # ==========================================
        # STEP 10: RETURN RESPONSE WITH CHARTS
        # ==========================================
        logger.info("Strategist report generated successfully")
        
        return {
            "success": True,
            "forecasts": forecasts,
            "weekly_forecast": weekly_data,
            "predicted_revenue": forecasts['30day']['total_revenue'],
            "revenue_range": {
                'min': forecasts['30day']['min_revenue'],
                'max': forecasts['30day']['max_revenue']
            },
            "restock_table": restock_data,
            "insights": insights,
            "last_data_date": actual_last_date.strftime('%Y-%m-%d'),
            "total_skus_analyzed": len(top_skus) if sku_col else 0,
            "data_period_days": (actual_last_date - df['ds'].min()).days,
            "revenue_change_percent": round(revenue_change, 1),
            # New chart data
            "weekday_chart": weekday_chart,
            "daily_orders_chart": daily_orders_chart,
            "daily_revenue_chart": daily_revenue_chart,
            "return_reasons_chart": return_reasons_chart
        }

    except Exception as e:
        logger.error(f"Strategist Engine Error: {e}", exc_info=True)
        return {"error": str(e)}