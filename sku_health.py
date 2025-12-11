"""
SKU Health Analysis Module

Provides comprehensive SKU health scoring using machine learning,
including feature engineering, predictive modeling, and anomaly detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import warnings
import gc

# At the end of analyze_sku_health function, add:
gc.collect()  # Force garbage collection to free memory

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    r2: float
    rmse: float
    mae: float
    
    def __str__(self):
        return f"R²: {self.r2:.4f}, RMSE: {self.rmse:.4f}, MAE: {self.mae:.4f}"


@dataclass
class SKUHealthResult:
    """Container for SKU health analysis results."""
    scored_df: pd.DataFrame
    model_obj: Dict
    train_metrics: ModelMetrics
    importances: pd.Series


class ColumnDetector:
    """Utility class for detecting column names with variations."""
    
    PRICE_VARIATIONS = [
        "item-price", "item_price", "item price",
        "amount", "selling price", "selling_price",
        "sale-price", "sale price", "total"
    ]
    
    RETURN_QTY_VARIATIONS = [
        "return quantity", "returned qty", "qty", "quantity"
    ]
    
    SKU_VARIATIONS = [
        "merchant sku", "sku", "product-sku", "product sku"
    ]
    
    @staticmethod
    def find_column(df: pd.DataFrame, variations: List[str]) -> Optional[str]:
        """
        Find a column in the dataframe matching one of the variations.
        
        Args:
            df: DataFrame to search
            variations: List of possible column name variations
            
        Returns:
            Actual column name if found, None otherwise
        """
        df_cols_lower = {c.lower().strip(): c for c in df.columns}
        
        for var in variations:
            normalized = var.lower().strip()
            if normalized in df_cols_lower:
                return df_cols_lower[normalized]
        
        return None


class SKUFeatureEngineer:
    """Handles feature engineering for SKU analysis."""
    
    def __init__(self, merged_data: pd.DataFrame, returns: pd.DataFrame, 
                 unpaid_orders: Optional[pd.DataFrame] = None):
        """
        Initialize the feature engineer.
        
        Args:
            merged_data: Main orders data
            returns: Returns data
            unpaid_orders: Unpaid orders data (optional)
        """
        self.df = merged_data.copy()
        self.returns = returns.copy()
        self.unpaid_orders = unpaid_orders.copy() if unpaid_orders is not None else None
        self._standardize_data()
        
    def _standardize_data(self):
        """Standardize column names and data types."""
        # Standardize SKU
        if "sku" in self.df.columns:
            self.df["sku"] = self.df["sku"].astype(str).str.strip().str.upper()
        
        # Ensure quantity is numeric
        self.df["quantity"] = pd.to_numeric(
            self.df.get("quantity", 0), errors="coerce"
        ).fillna(0)
        
        # Handle price column
        price_col = ColumnDetector.find_column(self.df, ColumnDetector.PRICE_VARIATIONS)
        if price_col:
            self.df["item_price"] = pd.to_numeric(
                self.df[price_col], errors="coerce"
            ).fillna(0)
        else:
            logger.warning("No price column found, defaulting to 0")
            self.df["item_price"] = 0
        
        # Calculate total revenue
        if "total" in self.df.columns:
            self.df["total"] = pd.to_numeric(
                self.df["total"], errors="coerce"
            ).fillna(self.df["quantity"] * self.df["item_price"])
        else:
            self.df["total"] = self.df["quantity"] * self.df["item_price"]
        
        # Standardize cost columns
        self.df["Product Cost"] = pd.to_numeric(
            self.df.get("Product Cost", 0), errors="coerce"
        ).fillna(0)
        self.df["Amz Fees"] = pd.to_numeric(
            self.df.get("Amz Fees", 0), errors="coerce"
        ).fillna(0)
        
        # Parse dates
        self.df["order_date"] = pd.to_datetime(
            self.df.get("order_date"), errors="coerce"
        )
    
    def _calculate_basic_metrics(self) -> pd.DataFrame:
        """Calculate basic aggregated metrics per SKU."""
        # First, ensure we have order IDs
        if "amazon-order-id" not in self.df.columns:
            logger.warning("No 'amazon-order-id' column found, using row count for orders")
            self.df["amazon-order-id"] = range(len(self.df))
        
        # Aggregate by SKU
        agg_dict = {
            "quantity": "sum",
            "total": "sum",
            "item_price": "mean",
            "Product Cost": "mean",
            "Amz Fees": "mean",
        }
        
        # Handle order ID carefully
        if "amazon-order-id" in self.df.columns:
            agg_dict["amazon-order-id"] = ["nunique", "count"]
        
        grouped = self.df.groupby("sku", as_index=True).agg(agg_dict)
        
        # Flatten multi-level columns if they exist
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in grouped.columns.values]
        
        # Rename to standard names
        rename_dict = {
            "quantity": "total_qty",
            "quantity_sum": "total_qty",
            "total": "total_revenue",
            "total_sum": "total_revenue",
            "item_price": "avg_price",
            "item_price_mean": "avg_price",
            "Product Cost": "avg_product_cost",
            "Product Cost_mean": "avg_product_cost",
            "Amz Fees": "avg_amz_fees",
            "Amz Fees_mean": "avg_amz_fees",
            "amazon-order-id_nunique": "total_orders",
            "amazon-order-id_count": "order_count",
        }
        
        grouped = grouped.rename(columns=rename_dict)
        
        # Ensure required columns exist
        if "total_orders" not in grouped.columns:
            grouped["total_orders"] = grouped.get("order_count", 1)
        if "order_count" not in grouped.columns:
            grouped["order_count"] = grouped.get("total_orders", 1)
            
        return grouped
    
    def _calculate_profitability(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
        """Add profitability metrics."""
        sku_grp["total_cost"] = sku_grp["total_qty"] * sku_grp["avg_product_cost"]
        sku_grp["total_fees"] = sku_grp["total_qty"] * sku_grp["avg_amz_fees"]
        sku_grp["profit"] = (
            sku_grp["total_revenue"] - sku_grp["total_cost"] - sku_grp["total_fees"]
        )
        
        # Avoid division by zero
        revenue_safe = sku_grp["total_revenue"].replace(0, np.nan)
        sku_grp["profit_margin"] = (sku_grp["profit"] / revenue_safe).fillna(0)
        
        return sku_grp
    
    def _calculate_temporal_features(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Calculate date span
        valid_dates = self.df["order_date"].dropna()
        if len(valid_dates) > 0:
            span_days = (valid_dates.max() - valid_dates.min()).days + 1
            span_days = max(span_days, 1)
        else:
            logger.warning("No valid dates found, using 30-day default")
            span_days = 30
        
        sku_grp["sales_velocity_per_day"] = sku_grp["total_qty"] / span_days
        
        # Recency
        last_sale_date = self.df.groupby("sku")["order_date"].max()
        days_since = (pd.Timestamp.now().normalize() - last_sale_date).dt.days
        sku_grp["days_since_last_sale"] = days_since.fillna(9999)
        
        return sku_grp
    
    def _calculate_returns_metrics(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
        """Add return-related metrics."""
        return_qty_col = ColumnDetector.find_column(
            self.returns, ColumnDetector.RETURN_QTY_VARIATIONS
        )
        sku_col = ColumnDetector.find_column(
            self.returns, ColumnDetector.SKU_VARIATIONS
        )
        
        if return_qty_col and sku_col:
            self.returns[sku_col] = (
                self.returns[sku_col].astype(str).str.strip().str.upper()
            )
            self.returns[return_qty_col] = pd.to_numeric(
                self.returns[return_qty_col], errors="coerce"
            ).fillna(0)
            
            returns_by_sku = self.returns.groupby(sku_col)[return_qty_col].sum()
            sku_grp["total_returns"] = sku_grp.index.map(returns_by_sku).fillna(0)
        else:
            logger.warning("Could not find return columns, defaulting to 0")
            sku_grp["total_returns"] = 0
        
        # Calculate return rate
        qty_safe = sku_grp["total_qty"].replace(0, np.nan)
        sku_grp["return_rate"] = (sku_grp["total_returns"] / qty_safe).fillna(0)
        
        return sku_grp
    
    def _calculate_unpaid_metrics(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
        """Add unpaid order metrics."""
        if self.unpaid_orders is None or self.unpaid_orders.empty:
            sku_grp["unpaid_qty"] = 0
            sku_grp["unpaid_ratio"] = 0
            return sku_grp
        
        if "SKU" in self.unpaid_orders.columns and "Quantity" in self.unpaid_orders.columns:
            self.unpaid_orders["SKU"] = (
                self.unpaid_orders["SKU"].astype(str).str.strip().str.upper()
            )
            self.unpaid_orders["Quantity"] = pd.to_numeric(
                self.unpaid_orders["Quantity"], errors="coerce"
            ).fillna(0)
            
            unpaid_by_sku = self.unpaid_orders.groupby("SKU")["Quantity"].sum()
            sku_grp["unpaid_qty"] = sku_grp.index.map(unpaid_by_sku).fillna(0)
        else:
            logger.warning("Unpaid orders missing required columns")
            sku_grp["unpaid_qty"] = 0
        
        # Calculate unpaid ratio
        qty_safe = sku_grp["total_qty"].replace(0, np.nan)
        sku_grp["unpaid_ratio"] = (sku_grp["unpaid_qty"] / qty_safe).fillna(0)
        
        return sku_grp
    
    def _calculate_stability_features(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
        """Add stability and volatility metrics."""
        # Geographic stability
        if "ship_state" in self.df.columns:
            state_pivot = self.df.groupby(
                ["sku", "ship_state"]
            )["quantity"].sum().unstack(fill_value=0)
            sku_grp["state_qty_variance"] = state_pivot.var(axis=1).fillna(0)
        else:
            sku_grp["state_qty_variance"] = 0
        
        # Weekly volatility
        if self.df["order_date"].notna().sum() > 0:
            self.df["week"] = self.df["order_date"].dt.to_period("W").apply(
                lambda r: r.start_time
            )
            weekly_pivot = self.df.groupby(
                ["sku", "week"]
            )["quantity"].sum().unstack(fill_value=0)
            sku_grp["weekly_volatility"] = weekly_pivot.std(axis=1).fillna(0)
        else:
            sku_grp["weekly_volatility"] = 0
        
        return sku_grp
    
    def build_features(self) -> pd.DataFrame:
        """
        Build complete feature set for SKU analysis.
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Building SKU features...")
        
        sku_grp = self._calculate_basic_metrics()
        sku_grp = self._calculate_profitability(sku_grp)
        sku_grp = self._calculate_temporal_features(sku_grp)
        sku_grp = self._calculate_returns_metrics(sku_grp)
        sku_grp = self._calculate_unpaid_metrics(sku_grp)
        sku_grp = self._calculate_stability_features(sku_grp)
        
        sku_grp = sku_grp.fillna(0)
        sku_grp.index.name = "SKU"
        
        logger.info(f"Built features for {len(sku_grp)} SKUs")
        return sku_grp


class SKUHealthScorer:
    """Handles SKU health scoring and modeling."""
    
    # Feature weights for proxy label
    PROXY_WEIGHTS = {
        "sales_velocity": 0.30,      # Increased - reward high volume
        "total_volume": 0.15,         # NEW - penalize low quantity
        "return_quality": 0.20,       # Slightly reduced
        "profit_margin": 0.20,        # Slightly reduced
        "payment_quality": 0.05,      # Reduced
        "geographic_stability": 0.05, # Reduced
        "temporal_stability": 0.05    # Reduced
    }
    
    def __init__(self, n_estimators: int = 50, test_size: float = 0.2, 
                 random_state: int = 42):
        """
        Initialize the scorer.
        
        Args:
            n_estimators: Number of trees in random forest (reduced for production)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        
    @staticmethod
    def _normalize_series(series: pd.Series, clip_range: Optional[Tuple[float, float]] = None) -> pd.Series:
        """
        Normalize a series to 0-100 scale.
        
        Args:
            series: Series to normalize
            clip_range: Optional tuple to clip values before normalization
            
        Returns:
            Normalized series
        """
        s = series.copy()
        
        if clip_range:
            s = s.clip(*clip_range)
        
        min_val, max_val = s.min(), s.max()
        
        if max_val == min_val:
            return pd.Series(50.0, index=s.index)
        
        return 100 * (s - min_val) / (max_val - min_val)
    
    def compute_proxy_label(self, sku_features: pd.DataFrame) -> pd.Series:
        """
        Compute proxy health score based on weighted features.
        
        Args:
            sku_features: DataFrame with SKU features
            
        Returns:
            Series with proxy health scores (0-100)
        """
        w = self.PROXY_WEIGHTS
        
        # Apply log transformation to total quantity to handle scale differences
        # This gives diminishing returns to very high volumes while still rewarding growth
        log_qty = np.log1p(sku_features["total_qty"])  # log1p avoids log(0)
        
        # Individual components
        velocity_score = w["sales_velocity"] * self._normalize_series(
            sku_features["sales_velocity_per_day"]
        )
        
        # NEW: Volume score - heavily penalizes low-volume SKUs
        volume_score = w["total_volume"] * self._normalize_series(log_qty)
        
        return_score = w["return_quality"] * (
            100 * (1 - sku_features["return_rate"])
        )
        
        profit_score = w["profit_margin"] * self._normalize_series(
            sku_features["profit_margin"], clip_range=(-1, 1)
        )
        
        payment_score = w["payment_quality"] * (
            100 * (1 - sku_features["unpaid_ratio"])
        )
        
        geo_stability_score = w["geographic_stability"] * (
            100 - self._normalize_series(sku_features["state_qty_variance"])
        )
        
        temporal_stability_score = w["temporal_stability"] * (
            100 - self._normalize_series(sku_features["weekly_volatility"])
        )
        
        # Combine scores
        total_score = (
            velocity_score + volume_score + return_score + profit_score + 
            payment_score + geo_stability_score + temporal_stability_score
        )
        
        # Apply minimum volume threshold penalty
        # SKUs with less than 5 units sold get a significant penalty
        min_volume_penalty = np.where(
            sku_features["total_qty"] < 5,
            0.5,  # 50% penalty
            1.0   # No penalty
        )
        
        total_score = total_score * min_volume_penalty
        
        # Final normalization to 0-100
        return self._normalize_series(total_score)
    
    def train_model(self, sku_features: pd.DataFrame, 
                   proxy_label: pd.Series) -> Dict:
        """
        Train random forest model for health prediction.
        
        Args:
            sku_features: Feature DataFrame
            proxy_label: Target labels
            
        Returns:
            Dictionary containing model, scaler, and metadata
        """
        logger.info("Training SKU health model...")
        
        # Prepare data
        X = sku_features.select_dtypes(include=[np.number]).fillna(0)
        y = proxy_label.loc[X.index].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model (n_jobs=1 for Windows stability)
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
            max_depth=10,  # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = ModelMetrics(
            r2=r2_score(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            mae=mean_absolute_error(y_test, y_pred)
        )
        
        logger.info(f"Model trained: {metrics}")
        
        # Feature importance
        imp = permutation_importance(
            model, X_test_scaled, y_test, 
            n_repeats=10, random_state=self.random_state, n_jobs=1
        )
        importance_series = pd.Series(
            imp.importances_mean, index=X.columns
        ).sort_values(ascending=False)
        
        return {
            "model": model,
            "scaler": scaler,
            "features": list(X.columns),
            "metrics": metrics,
            "importances": importance_series
        }
    
    def score_skus(self, model_obj: Dict, sku_features: pd.DataFrame, 
                   n_clusters: int = 3) -> pd.DataFrame:
        """
        Apply model to score all SKUs and add clustering/anomaly detection.
        
        Args:
            model_obj: Trained model object
            sku_features: Features to score
            n_clusters: Number of clusters for segmentation
            
        Returns:
            DataFrame with predictions and additional analysis
        """
        logger.info("Scoring SKUs...")
        
        # Prepare features
        X = sku_features.select_dtypes(include=[np.number])
        X = X[model_obj["features"]]
        X_scaled = model_obj["scaler"].transform(X)
        
        # Predict
        predictions = model_obj["model"].predict(X_scaled)
        
        # Build output
        output = sku_features.copy()
        output["health_score_pred"] = predictions
        
        # Clustering for segmentation
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        output["cluster"] = kmeans.fit_predict(X_scaled)
        
        # Anomaly detection
        iso_forest = IsolationForest(
            n_estimators=150, 
            contamination=0.05,  # Slightly higher for better detection
            random_state=self.random_state
        )
        output["anomaly"] = iso_forest.fit_predict(X_scaled) == -1
        
        # Add key issues detection
        output["key_issues"] = output.apply(self._identify_key_issues, axis=1)
        
        logger.info(f"Scored {len(output)} SKUs, found {output['anomaly'].sum()} anomalies")
        
        return output
    
    @staticmethod
    def _identify_key_issues(row: pd.Series) -> str:
        """
        Identify key issues for a SKU based on its metrics.
        
        Issues detected:
        - Low Volume: Less than 10 units sold total
        - High Returns: Return rate > 15%
        - Moderate Returns: Return rate > 8%
        - Negative Margin: Selling at a loss
        - Low Margin: Profit margin < 10%
        - Payment Issues: >10% of orders unpaid
        - Slow Moving: Less than 0.1 units sold per day
        - Stale Inventory: No sales in 90+ days
        - Inactive: No sales in 60+ days
        - Volatile Demand: High weekly sales fluctuation
        - Geographic Risk: Sales concentrated in few states
        - Anomaly Detected: AI flagged unusual pattern
        - Few Orders: Less than 5 total orders
        
        Args:
            row: Series containing SKU metrics
            
        Returns:
            Comma-separated string of issues, or "None" if healthy
        """
        issues = []
        
        # Low sales volume
        if row.get("total_qty", 0) < 10:
            issues.append("Low Volume")
        
        # High return rate
        if row.get("return_rate", 0) > 0.15:  # >15% returns
            issues.append("High Returns")
        elif row.get("return_rate", 0) > 0.08:  # >8% returns
            issues.append("Moderate Returns")
        
        # Poor profit margin
        if row.get("profit_margin", 0) < 0:
            issues.append("Negative Margin")
        elif row.get("profit_margin", 0) < 0.10:  # <10%
            issues.append("Low Margin")
        
        # Unpaid orders issue
        if row.get("unpaid_ratio", 0) > 0.10:  # >10% unpaid
            issues.append("Payment Issues")
        
        # Slow sales velocity
        if row.get("sales_velocity_per_day", 0) < 0.1:  # <0.1 units/day
            issues.append("Slow Moving")
        
        # Not sold recently
        if row.get("days_since_last_sale", 9999) > 90:
            issues.append("Stale Inventory")
        elif row.get("days_since_last_sale", 9999) > 60:
            issues.append("Inactive")
        
        # High volatility
        if row.get("weekly_volatility", 0) > row.get("total_qty", 1) * 0.5:
            issues.append("Volatile Demand")
        
        # Geographic concentration issues
        if row.get("state_qty_variance", 0) > row.get("total_qty", 1) * 2:
            issues.append("Geographic Risk")
        
        # Anomaly flag
        if row.get("anomaly", False):
            issues.append("Anomaly Detected")
        
        # Low order count
        if row.get("total_orders", 0) < 5:
            issues.append("Few Orders")
        
        return ", ".join(issues) if issues else "None"


def analyze_sku_health(merged_data: pd.DataFrame, 
                      returns: pd.DataFrame,
                      unpaid_orders: Optional[pd.DataFrame] = None,
                      n_estimators: int = 200,
                      test_size: float = 0.2,
                      random_state: int = 42,
                      min_orders: int = 1) -> SKUHealthResult:
    """
    Complete pipeline for SKU health analysis.
    
    Args:
        merged_data: Main orders data
        returns: Returns data
        unpaid_orders: Unpaid orders data (optional)
        n_estimators: Number of trees in random forest
        test_size: Test set proportion
        random_state: Random seed
        min_orders: Minimum number of orders to include SKU (default: 1)
        
    Returns:
        SKUHealthResult object with complete analysis
    """
    logger.info("Starting SKU health analysis pipeline...")
    logger.info(f"Input data shape: {merged_data.shape}")
    
    # Feature engineering
    engineer = SKUFeatureEngineer(merged_data, returns, unpaid_orders)
    sku_features = engineer.build_features()
    
    # Log statistics before filtering
    logger.info(f"Total unique SKUs before filtering: {len(sku_features)}")
    logger.info(f"Order count distribution:\n{sku_features['total_orders'].describe()}")
    
    # Filter out low-volume SKUs
    original_count = len(sku_features)
    sku_features = sku_features[sku_features['total_orders'] >= min_orders]
    filtered_count = len(sku_features)
    
    logger.info(f"SKUs after filtering (min {min_orders} orders): {filtered_count}")
    logger.info(f"Filtered out: {original_count - filtered_count} SKUs")
    
    if len(sku_features) == 0:
        logger.error("No SKUs remain after filtering!")
        raise ValueError(f"No SKUs have at least {min_orders} orders. Try lowering min_orders parameter.")
    
    # Log top SKUs by quantity
    top_skus = sku_features.nlargest(5, 'total_qty')[['total_qty', 'total_orders', 'profit_margin']]
    logger.info(f"Top 5 SKUs by quantity:\n{top_skus}")
    
    # Scoring and modeling
    scorer = SKUHealthScorer(n_estimators, test_size, random_state)
    proxy_label = scorer.compute_proxy_label(sku_features)
    model_obj = scorer.train_model(sku_features, proxy_label)
    scored_df = scorer.score_skus(model_obj, sku_features)
    
    # Add proxy label for comparison
    scored_df["proxy_label"] = proxy_label
    
    # Sort by health score (descending - best first)
    scored_df = scored_df.reset_index().sort_values("health_score_pred", ascending=False)
    
    logger.info("SKU health analysis complete!")
    logger.info(f"Score range: {scored_df['health_score_pred'].min():.2f} - {scored_df['health_score_pred'].max():.2f}")
    
    # Convert metrics dataclass → dict so Flask can safely read it
    metrics_dict = {
        "r2": model_obj["metrics"].r2,
        "rmse": model_obj["metrics"].rmse,
        "mae": model_obj["metrics"].mae
    }

    return SKUHealthResult(
        scored_df=scored_df,
        model_obj=model_obj,
        train_metrics=metrics_dict,
        importances=model_obj["importances"]
    )



# Maintain backward compatibility with original function name
def build_and_score_sku_health(merged_data: pd.DataFrame,
                               Return: pd.DataFrame,
                               unpaid_orders: Optional[pd.DataFrame] = None,
                               output_dir: Optional[str] = None) -> Dict:
    """
    Legacy function name for backward compatibility.
    
    Args:
        merged_data: Main orders data
        Return: Returns data
        unpaid_orders: Unpaid orders data
        output_dir: Unused parameter (kept for compatibility)
        
    Returns:
        Dictionary with analysis results
    """
    result = analyze_sku_health(merged_data, Return, unpaid_orders)
    
    return {
        "scored_df": result.scored_df,
        "model_obj": result.model_obj,
        "train_metrics": result.train_metrics,
        "importances": result.importances
    }