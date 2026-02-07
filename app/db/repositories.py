import uuid
from datetime import datetime
import pandas as pd
import logging
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from app.db.connection import database as db
from app.db.models import (
    Analysis,
    SummaryMetric,
    TopSKU,
    TopReturn,
    StateSales,
    Transaction
)

logger = logging.getLogger(__name__)

def parse_date(date_val):
    """Helper to safely convert strings/timestamps to python date objects."""
    if pd.isna(date_val) or date_val == "" or date_val is None:
        return None
    if isinstance(date_val, str):
        try:
            return datetime.strptime(date_val, '%Y-%m-%d').date()
        except ValueError:
            return None
    if isinstance(date_val, datetime):
        return date_val.date()
    return date_val


def get_or_create_analysis(user_id: int,file_name: str, start_date, end_date, summary: dict):
    """
    Get existing analysis or create new one based on date range.
    Returns (analysis_id, is_new_analysis)
    """
    p_start_date = parse_date(start_date)
    p_end_date = parse_date(end_date)
    
    # Check for existing analysis with same date range
    existing = Analysis.query.filter_by(
        user_id=user_id,
        start_date=p_start_date,
        end_date=p_end_date
    ).first()
    
    if existing:
        logger.info(f"📊 Found existing Analysis {existing.id} for date range {p_start_date} to {p_end_date}")
        # Update the analysis_data JSON blob
        existing.analysis_data = summary
        existing.file_name = file_name
        db.session.flush()
        return existing.id, False
    else:
        logger.info(f"✨ Creating NEW Analysis for date range {p_start_date} to {p_end_date}")
        new_analysis = Analysis(
            id=uuid.uuid4(),
            user_id=user_id,
            file_name=file_name,
            start_date=p_start_date,
            end_date=p_end_date,
            analysis_data=summary
        )
        db.session.add(new_analysis)
        db.session.flush()
        return new_analysis.id, True


def insert_transactions_incrementally(analysis_id, merged_data: list):
    """
    Insert only NEW transactions matching the 14-column report structure.
    """
    new_count = 0
    skipped_count = 0
    new_transactions = []
    
    for row in merged_data:
        order_id = row.get("amazon_order_id") or row.get("amazon-order-id")
        sku = row.get("sku")
        
        # Skip if key fields are missing
        if not order_id or not sku:
            skipped_count += 1
            continue
        
        # Check for duplicates
        exists = Transaction.query.filter(
            and_(
                Transaction.amazon_order_id == order_id,
                Transaction.sku == sku,
                Transaction.analysis_id == analysis_id
            )
        ).first()
        
        if exists:
            skipped_count += 1
            continue
        
        # --- MAPPING 14 COLUMNS TO DB ---
        new_txn = Transaction(
            analysis_id=analysis_id,
            amazon_order_id=order_id,
            sku=sku,
            
            # 1. Dates & Qty
            order_date=parse_date(row.get("order_date")),
            quantity=row.get("quantity"),
            
            # 2. Money Fields (Handling potential NaNs with 'or 0.0')
            item_price=row.get("item_price") or row.get("item-price") or 0.0,
            revenue=row.get("revenue") or 0.0,
            payment_amount=row.get("Payment Amount") or row.get("payment_amount") or 0.0,
            
            # 3. Costs & Fees
            product_cost=row.get("Product Cost") or row.get("product_cost") or 0.0,
            total_cost=row.get("Total Cost") or row.get("total_cost") or 0.0,
            amz_fees=row.get("Amz Fees") or row.get("amz_fees") or 0.0,
            
            # 4. Status & Details
            status=row.get("Status") or row.get("status"),
            ship_state=row.get("ship_state"),
            is_paid=str(row.get("Is Paid") or row.get("is_paid")),  # Convert boolean to string if needed
            return_reason=row.get("raw_return_reason") or row.get("return_reason")
        )
        
        db.session.add(new_txn)
        new_transactions.append(row)
        new_count += 1
    
    logger.info(f"📝 Transactions: {new_count} new, {skipped_count} skipped")
    return new_count, skipped_count, new_transactions


def update_summary_metrics_incrementally(analysis_id, summary: dict, is_new_analysis: bool):
    """
    Update summary metrics incrementally.
    If new analysis: create new metrics
    If existing: add delta to existing values
    """
    for metric, value in summary.items():
        numeric_value = float(value) if value else 0.0
        
        if is_new_analysis:
            # Create new metric
            db.session.add(SummaryMetric(
                analysis_id=analysis_id,
                metric=metric,
                value=numeric_value
            ))
        else:
            # Update existing metric incrementally
            existing_metric = SummaryMetric.query.filter_by(
                analysis_id=analysis_id,
                metric=metric
            ).first()
            
            if existing_metric:
                existing_metric.value += numeric_value
                logger.debug(f"📊 Updated {metric}: added {numeric_value} to existing value")
            else:
                # Metric doesn't exist yet, create it
                db.session.add(SummaryMetric(
                    analysis_id=analysis_id,
                    metric=metric,
                    value=numeric_value
                ))
    
    logger.info(f"✅ Summary metrics {'created' if is_new_analysis else 'updated incrementally'}")


def update_top_skus_incrementally(analysis_id, top_skus: list, is_new_analysis: bool):
    """
    Update top SKUs incrementally by aggregating quantities.
    """
    if is_new_analysis:
        # Create new records
        for i, row in enumerate(top_skus, start=1):
            db.session.add(TopSKU(
                analysis_id=analysis_id,
                rank=i,
                sku=row.get("sku"),
                quantity=row.get("quantity", 0)
            ))
    else:
        # Aggregate with existing records
        for row in top_skus:
            sku = row.get("sku")
            qty = row.get("quantity", 0)
            
            existing = TopSKU.query.filter_by(
                analysis_id=analysis_id,
                sku=sku
            ).first()
            
            if existing:
                existing.quantity += qty
            else:
                db.session.add(TopSKU(
                    analysis_id=analysis_id,
                    rank=None,  # Will be recalculated
                    sku=sku,
                    quantity=qty
                ))
        
        # Recalculate ranks
        all_skus = TopSKU.query.filter_by(analysis_id=analysis_id).order_by(TopSKU.quantity.desc()).all()
        for i, sku_obj in enumerate(all_skus, start=1):
            sku_obj.rank = i
    
    logger.info(f"✅ Top SKUs {'created' if is_new_analysis else 'updated incrementally'}")


def update_top_returns_incrementally(analysis_id, top_returns: list, is_new_analysis: bool):
    """
    Update top returns incrementally by aggregating quantities.
    """
    if is_new_analysis:
        for i, row in enumerate(top_returns, start=1):
            db.session.add(TopReturn(
                analysis_id=analysis_id,
                rank=i,
                sku=row.get("sku"),
                quantity=row.get("quantity", 0),
                category=row.get("category")
            ))
    else:
        for row in top_returns:
            sku = row.get("sku")
            qty = row.get("quantity", 0)
            category = row.get("category")
            
            existing = TopReturn.query.filter_by(
                analysis_id=analysis_id,
                sku=sku
            ).first()
            
            if existing:
                existing.quantity += qty
                existing.category = category  # Update category if changed
            else:
                db.session.add(TopReturn(
                    analysis_id=analysis_id,
                    rank=None,
                    sku=sku,
                    quantity=qty,
                    category=category
                ))
        
        # Recalculate ranks
        all_returns = TopReturn.query.filter_by(analysis_id=analysis_id).order_by(TopReturn.quantity.desc()).all()
        for i, return_obj in enumerate(all_returns, start=1):
            return_obj.rank = i
    
    logger.info(f"✅ Top Returns {'created' if is_new_analysis else 'updated incrementally'}")


def update_state_sales_incrementally(analysis_id, top_states: list, is_new_analysis: bool):
    """
    Update state sales incrementally by aggregating quantities.
    """
    if is_new_analysis:
        for i, row in enumerate(top_states, start=1):
            db.session.add(StateSales(
                analysis_id=analysis_id,
                rank=i,
                state=row.get("state"),
                quantity=row.get("total_orders", 0)
            ))
    else:
        for row in top_states:
            state = row.get("state")
            qty = row.get("total_orders", 0)
            
            existing = StateSales.query.filter_by(
                analysis_id=analysis_id,
                state=state
            ).first()
            
            if existing:
                existing.quantity += qty
            else:
                db.session.add(StateSales(
                    analysis_id=analysis_id,
                    rank=None,
                    state=state,
                    quantity=qty
                ))
        
        # Recalculate ranks
        all_states = StateSales.query.filter_by(analysis_id=analysis_id).order_by(StateSales.quantity.desc()).all()
        for i, state_obj in enumerate(all_states, start=1):
            state_obj.rank = i
    
    logger.info(f"✅ State Sales {'created' if is_new_analysis else 'updated incrementally'}")


def save_full_analysis(
    *,
    user_id: int,
    file_name: str,
    start_date,
    end_date,
    summary: dict,
    top_skus: list,
    top_returns: list,
    top_states: list,
    merged_data: list
):
    """
    Persists analysis data with incremental updates.
    
    - Uses composite primary key (amazon_order_id, sku) for transactions
    - Skips duplicate transactions silently
    - Updates summaries cumulatively instead of overwriting
    """
    try:
        # 1. Get or create analysis
        analysis_id, is_new_analysis = get_or_create_analysis(
            user_id, file_name, start_date, end_date, summary
        )
        
        # 2. Insert transactions incrementally (skip duplicates)
        new_count, skipped_count, new_transactions = insert_transactions_incrementally(
            analysis_id, merged_data
        )
        
        # 3. Update summary tables incrementally (only if new transactions were added)
        if new_count > 0 or is_new_analysis:
            update_summary_metrics_incrementally(analysis_id, summary, is_new_analysis)
            update_top_skus_incrementally(analysis_id, top_skus, is_new_analysis)
            update_top_returns_incrementally(analysis_id, top_returns, is_new_analysis)
            update_state_sales_incrementally(analysis_id, top_states, is_new_analysis)
        else:
            logger.info("ℹ️ No new transactions added, summaries unchanged")
        
        # 4. Commit all changes
        db.session.commit()
        
        logger.info(f"✅ Analysis saved successfully: {analysis_id}")
        logger.info(f"📊 Summary: {new_count} new transactions, {skipped_count} duplicates skipped")
        
        return str(analysis_id)
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Error saving analysis: {str(e)}", exc_info=True)
        raise