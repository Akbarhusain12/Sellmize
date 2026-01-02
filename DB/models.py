import uuid
from datetime import datetime
from DB.db import database as db
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy import UniqueConstraint

SCHEMA = "sellmize"

class Analysis(db.Model):
    __tablename__ = "analyses"
    __table_args__ = {"schema": SCHEMA}

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    file_name = db.Column(db.String)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)

    analysis_data = db.Column(JSON)

    # Relationships
    summary_metrics = db.relationship('SummaryMetric', backref='analysis', lazy=True, cascade="all, delete-orphan")
    top_skus = db.relationship('TopSKU', backref='analysis', lazy=True, cascade="all, delete-orphan")
    top_returns = db.relationship('TopReturn', backref='analysis', lazy=True, cascade="all, delete-orphan")
    state_sales = db.relationship('StateSales', backref='analysis', lazy=True, cascade="all, delete-orphan")
    transactions = db.relationship('Transaction', backref='analysis', lazy=True, cascade="all, delete-orphan")


class Transaction(db.Model):
    __tablename__ = "transactions"
    __table_args__ = (
        db.PrimaryKeyConstraint('amazon_order_id', 'sku', 'analysis_id'),
        {"schema": SCHEMA}
    )

    # Composite primary key fields
    amazon_order_id = db.Column(db.String, nullable=False)
    sku = db.Column(db.String, nullable=False)
    
    # Foreign key (also part of composite PK to ensure referential integrity)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey(f"{SCHEMA}.analyses.id"),
        nullable=False
    )
    
    # Other fields
    order_date = db.Column(db.Date)
    quantity = db.Column(db.Integer)
    total_amount = db.Column(db.Float)
    product_cost = db.Column(db.Float)
    total_cost = db.Column(db.Float)
    status = db.Column(db.String)
    ship_state = db.Column(db.String)
    amz_fees = db.Column(db.Float)


class SummaryMetric(db.Model):
    __tablename__ = "summary_metrics"
    __table_args__ = (
        UniqueConstraint('analysis_id', 'metric', name='uq_analysis_metric'),
        {"schema": SCHEMA}
    )

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey(f"{SCHEMA}.analyses.id"),
        nullable=False
    )

    metric = db.Column(db.String, nullable=False)
    value = db.Column(db.Float, nullable=False)


class TopSKU(db.Model):
    __tablename__ = "top_skus"
    __table_args__ = {"schema": SCHEMA}

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey(f"{SCHEMA}.analyses.id"),
        nullable=False
    )

    rank = db.Column(db.Integer)
    sku = db.Column(db.String)
    quantity = db.Column(db.Integer)


class TopReturn(db.Model):
    __tablename__ = "top_returns"
    __table_args__ = {"schema": SCHEMA}

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey(f"{SCHEMA}.analyses.id"),
        nullable=False
    )

    rank = db.Column(db.Integer)
    sku = db.Column(db.String)
    quantity = db.Column(db.Integer)
    category = db.Column(db.String)


class StateSales(db.Model):
    __tablename__ = "state_sales"
    __table_args__ = {"schema": SCHEMA}

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey(f"{SCHEMA}.analyses.id"),
        nullable=False
    )

    rank = db.Column(db.Integer)
    state = db.Column(db.String)
    quantity = db.Column(db.Integer)