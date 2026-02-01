import uuid
from datetime import datetime
from app.db.connection import database as db
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy import UniqueConstraint
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now())
    
    # Relationship to analysis
    analyses = db.relationship('Analysis', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
        
class Analysis(db.Model):
    __tablename__ = "analyses"
    

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
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
    )

    # Composite primary key fields
    amazon_order_id = db.Column(db.String, nullable=False)
    sku = db.Column(db.String, nullable=False)
    
    # Foreign key (also part of composite PK to ensure referential integrity)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("analyses.id"),
        nullable=False
    )
    
    # Other fields
    order_date = db.Column(db.Date)
    quantity = db.Column(db.Integer)
    total_amount = db.Column(db.Float)
    real_revenue = db.Column(db.Float)
    product_cost = db.Column(db.Float)
    total_cost = db.Column(db.Float)
    status = db.Column(db.String)
    ship_state = db.Column(db.String)
    amz_fees = db.Column(db.Float)


class SummaryMetric(db.Model):
    __tablename__ = "summary_metrics"
    __table_args__ = (
        UniqueConstraint('analysis_id', 'metric', name='uq_analysis_metric'),
    )

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("analyses.id"),
        nullable=False
    )

    metric = db.Column(db.String, nullable=False)
    value = db.Column(db.Float, nullable=False)


class TopSKU(db.Model):
    __tablename__ = "top_skus"
    

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("analyses.id"),
        nullable=False
    )

    rank = db.Column(db.Integer)
    sku = db.Column(db.String)
    quantity = db.Column(db.Integer)


class TopReturn(db.Model):
    __tablename__ = "top_returns"
    

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("analyses.id"),
        nullable=False
    )

    rank = db.Column(db.Integer)
    sku = db.Column(db.String)
    quantity = db.Column(db.Integer)
    category = db.Column(db.String)


class StateSales(db.Model):
    __tablename__ = "state_sales"
    

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("analyses.id"),
        nullable=False
    )

    rank = db.Column(db.Integer)
    state = db.Column(db.String)
    quantity = db.Column(db.Integer)
    
    
    
class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=True) # e.g., "Product Search for Nike"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to get all messages in this session
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade="all, delete-orphan")

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    sender = db.Column(db.String(50), nullable=False) 
    content = db.Column(db.Text, nullable=False)      
    meta_data = db.Column(JSON, nullable=True)       
    created_at = db.Column(db.DateTime, default=datetime.utcnow)