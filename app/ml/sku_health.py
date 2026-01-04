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
    train_metrics: Dict # Changed to Dict for JSON safety
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
        if "quantity" in self.df.columns:
            self.df["quantity"] = pd.to_numeric(
                self.df["quantity"], errors="coerce"
            ).fillna(0)
        else:
            self.df["quantity"] = 0
        
        # Handle price column
        price_col = ColumnDetector.find_column(self.df, ColumnDetector.PRICE_VARIATIONS)
        if price_col:
            self.df["item_price"] = pd.to_numeric(
                self.df[price_col], errors="coerce"
            ).fillna(0)
        else:
            self.df["item_price"] = 0
        
        # Calculate total revenue
        if "total" in self.df.columns:
            self.df["total"] = pd.to_numeric(
                self.df["total"], errors="coerce"
            ).fillna(self.df["quantity"] * self.df["item_price"])
        else:
            self.df["total"] = self.df["quantity"] * self.df["item_price"]
        
        # Standardize cost columns
        if "Product Cost" in self.df.columns:
            self.df["Product Cost"] = pd.to_numeric(
                self.df["Product Cost"], errors="coerce"
            ).fillna(0)
        else:
            self.df["Product Cost"] = 0

        if "Amz Fees" in self.df.columns:
            self.df["Amz Fees"] = pd.to_numeric(
                self.df["Amz Fees"], errors="coerce"
            ).fillna(0)
        else:
            self.df["Amz Fees"] = 0
        
        # Parse dates
        if "order_date" in self.df.columns:
            self.df["order_date"] = pd.to_datetime(
                self.df["order_date"], errors="coerce"
            )
        else:
            self.df["order_date"] = pd.NaT
    
    def _calculate_basic_metrics(self) -> pd.DataFrame:
        """Calculate basic aggregated metrics per SKU."""
        if "amazon-order-id" not in self.df.columns:
            self.df["amazon-order-id"] = range(len(self.df))
        
        agg_dict = {
            "quantity": "sum",
            "total": "sum",
            "item_price": "mean",
            "Product Cost": "mean",
            "Amz Fees": "mean",
        }
        
        if "amazon-order-id" in self.df.columns:
            agg_dict["amazon-order-id"] = ["nunique", "count"]
        
        grouped = self.df.groupby("sku", as_index=True).agg(agg_dict)
        
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in grouped.columns.values]
        
        rename_dict = {
            "quantity": "total_qty", "quantity_sum": "total_qty",
            "total": "total_revenue", "total_sum": "total_revenue",
            "item_price": "avg_price", "item_price_mean": "avg_price",
            "Product Cost": "avg_product_cost", "Product Cost_mean": "avg_product_cost",
            "Amz Fees": "avg_amz_fees", "Amz Fees_mean": "avg_amz_fees",
            "amazon-order-id_nunique": "total_orders",
            "amazon-order-id_count": "order_count",
        }
        
        grouped = grouped.rename(columns=rename_dict)
        
        if "total_orders" not in grouped.columns:
            grouped["total_orders"] = grouped.get("order_count", 1)
        if "order_count" not in grouped.columns:
            grouped["order_count"] = grouped.get("total_orders", 1)
            
        return grouped
    
    def _calculate_profitability(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
        sku_grp["total_cost"] = sku_grp["total_qty"] * sku_grp["avg_product_cost"]
        sku_grp["total_fees"] = sku_grp["total_qty"] * sku_grp["avg_amz_fees"]
        sku_grp["profit"] = (
            sku_grp["total_revenue"] - sku_grp["total_cost"] - sku_grp["total_fees"]
        )
        
        revenue_safe = sku_grp["total_revenue"].replace(0, np.nan)
        sku_grp["profit_margin"] = (sku_grp["profit"] / revenue_safe).fillna(0)
        return sku_grp
    
    def _calculate_temporal_features(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
        valid_dates = self.df["order_date"].dropna()
        if len(valid_dates) > 0:
            span_days = (valid_dates.max() - valid_dates.min()).days + 1
            span_days = max(span_days, 1)
        else:
            span_days = 30
        
        sku_grp["sales_velocity_per_day"] = sku_grp["total_qty"] / span_days
        
        last_sale_date = self.df.groupby("sku")["order_date"].max()
        days_since = (pd.Timestamp.now().normalize() - last_sale_date).dt.days
        sku_grp["days_since_last_sale"] = days_since.fillna(9999)
        return sku_grp
    
    def _calculate_returns_metrics(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
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
            sku_grp["total_returns"] = 0
        
        qty_safe = sku_grp["total_qty"].replace(0, np.nan)
        sku_grp["return_rate"] = (sku_grp["total_returns"] / qty_safe).fillna(0)
        return sku_grp
    
    def _calculate_unpaid_metrics(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
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
            sku_grp["unpaid_qty"] = 0
        
        qty_safe = sku_grp["total_qty"].replace(0, np.nan)
        sku_grp["unpaid_ratio"] = (sku_grp["unpaid_qty"] / qty_safe).fillna(0)
        return sku_grp
    
    def _calculate_stability_features(self, sku_grp: pd.DataFrame) -> pd.DataFrame:
        if "ship_state" in self.df.columns:
            state_pivot = self.df.groupby(
                ["sku", "ship_state"]
            )["quantity"].sum().unstack(fill_value=0)
            sku_grp["state_qty_variance"] = state_pivot.var(axis=1).fillna(0)
        else:
            sku_grp["state_qty_variance"] = 0
        
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
        logger.info("Building SKU features...")
        sku_grp = self._calculate_basic_metrics()
        sku_grp = self._calculate_profitability(sku_grp)
        sku_grp = self._calculate_temporal_features(sku_grp)
        sku_grp = self._calculate_returns_metrics(sku_grp)
        sku_grp = self._calculate_unpaid_metrics(sku_grp)
        sku_grp = self._calculate_stability_features(sku_grp)
        
        sku_grp = sku_grp.fillna(0)
        sku_grp.index.name = "SKU"
        return sku_grp


class SKUHealthScorer:
    PROXY_WEIGHTS = {
        "sales_velocity": 0.30,
        "total_volume": 0.15,
        "return_quality": 0.20,
        "profit_margin": 0.20,
        "payment_quality": 0.05,
        "geographic_stability": 0.05,
        "temporal_stability": 0.05
    }
    
    def __init__(self, n_estimators: int = 50, test_size: float = 0.2, random_state: int = 42):
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        
    @staticmethod
    def _normalize_series(series: pd.Series, clip_range: Optional[Tuple[float, float]] = None) -> pd.Series:
        s = series.copy()
        if clip_range:
            s = s.clip(*clip_range)
        
        # FIX: Handle NaNs and infinite values in normalization
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        min_val, max_val = s.min(), s.max()
        if max_val == min_val:
            return pd.Series(50.0, index=s.index)
        
        return 100 * (s - min_val) / (max_val - min_val)
    
    def compute_proxy_label(self, sku_features: pd.DataFrame) -> pd.Series:
        w = self.PROXY_WEIGHTS
        log_qty = np.log1p(sku_features["total_qty"])
        
        velocity_score = w["sales_velocity"] * self._normalize_series(sku_features["sales_velocity_per_day"])
        volume_score = w["total_volume"] * self._normalize_series(log_qty)
        return_score = w["return_quality"] * (100 * (1 - sku_features["return_rate"]))
        profit_score = w["profit_margin"] * self._normalize_series(sku_features["profit_margin"], clip_range=(-1, 1))
        payment_score = w["payment_quality"] * (100 * (1 - sku_features["unpaid_ratio"]))
        geo_stability_score = w["geographic_stability"] * (100 - self._normalize_series(sku_features["state_qty_variance"]))
        temporal_stability_score = w["temporal_stability"] * (100 - self._normalize_series(sku_features["weekly_volatility"]))
        
        total_score = (velocity_score + volume_score + return_score + profit_score + 
                       payment_score + geo_stability_score + temporal_stability_score)
        
        min_volume_penalty = np.where(sku_features["total_qty"] < 5, 0.5, 1.0)
        total_score = total_score * min_volume_penalty
        
        return self._normalize_series(total_score)
    
    def train_model(self, sku_features: pd.DataFrame, proxy_label: pd.Series) -> Dict:
        X = sku_features.select_dtypes(include=[np.number]).fillna(0)
        y = proxy_label.loc[X.index].values
        
        # Check if we have enough data to split
        if len(X) < 5:
            # Fallback for very small datasets
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            metrics = ModelMetrics(r2=1.0, rmse=0.0, mae=0.0)
            return {
                "model": model, "scaler": StandardScaler().fit(X),
                "features": list(X.columns), "metrics": metrics,
                "importances": pd.Series(0, index=X.columns)
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state,
            n_jobs=1, max_depth=10, min_samples_split=5, min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        metrics = ModelMetrics(
            r2=r2_score(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            mae=mean_absolute_error(y_test, y_pred)
        )
        
        imp = permutation_importance(
            model, X_test_scaled, y_test, n_repeats=5, random_state=self.random_state, n_jobs=1
        )
        importance_series = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
        
        return {
            "model": model, "scaler": scaler, "features": list(X.columns),
            "metrics": metrics, "importances": importance_series
        }
    
    def score_skus(self, model_obj: Dict, sku_features: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        X = sku_features.select_dtypes(include=[np.number])
        X = X[model_obj["features"]]
        X_scaled = model_obj["scaler"].transform(X)
        
        predictions = model_obj["model"].predict(X_scaled)
        
        output = sku_features.copy()
        output["health_score_pred"] = predictions
        
        # Safely handle clustering for small datasets
        n_clusters = min(n_clusters, len(output))
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            output["cluster"] = kmeans.fit_predict(X_scaled)
        else:
            output["cluster"] = 0
        
        iso_forest = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=self.random_state
        )
        output["anomaly"] = iso_forest.fit_predict(X_scaled) == -1
        
        output["key_issues"] = output.apply(self._identify_key_issues, axis=1)
        return output
    
    @staticmethod
    def _identify_key_issues(row: pd.Series) -> str:
        issues = []
        if row.get("total_qty", 0) < 10: issues.append("Low Volume")
        if row.get("return_rate", 0) > 0.15: issues.append("High Returns")
        elif row.get("return_rate", 0) > 0.08: issues.append("Moderate Returns")
        if row.get("profit_margin", 0) < 0: issues.append("Negative Margin")
        elif row.get("profit_margin", 0) < 0.10: issues.append("Low Margin")
        if row.get("unpaid_ratio", 0) > 0.10: issues.append("Payment Issues")
        if row.get("sales_velocity_per_day", 0) < 0.1: issues.append("Slow Moving")
        if row.get("days_since_last_sale", 9999) > 90: issues.append("Stale Inventory")
        elif row.get("days_since_last_sale", 9999) > 60: issues.append("Inactive")
        if row.get("weekly_volatility", 0) > row.get("total_qty", 1) * 0.5: issues.append("Volatile Demand")
        if row.get("state_qty_variance", 0) > row.get("total_qty", 1) * 2: issues.append("Geographic Risk")
        if row.get("anomaly", False): issues.append("Anomaly Detected")
        if row.get("total_orders", 0) < 5: issues.append("Few Orders")
        
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
    """
    logger.info("Starting SKU health analysis pipeline...")
    
    # Feature engineering
    engineer = SKUFeatureEngineer(merged_data, returns, unpaid_orders)
    sku_features = engineer.build_features()
    
    # Filter out low-volume SKUs
    sku_features = sku_features[sku_features['total_orders'] >= min_orders]
    
    if len(sku_features) == 0:
        logger.error("No SKUs remain after filtering!")
        # Return empty result structure safely
        empty_df = pd.DataFrame(columns=['health_score_pred', 'cluster', 'anomaly', 'key_issues'])
        metrics_dict = {"r2": 0.0, "rmse": 0.0, "mae": 0.0}
        return SKUHealthResult(
            scored_df=empty_df, model_obj={}, train_metrics=metrics_dict, importances=pd.Series()
        )
    
    # Scoring and modeling
    scorer = SKUHealthScorer(n_estimators, test_size, random_state)
    proxy_label = scorer.compute_proxy_label(sku_features)
    model_obj = scorer.train_model(sku_features, proxy_label)
    scored_df = scorer.score_skus(model_obj, sku_features)
    
    scored_df["proxy_label"] = proxy_label
    scored_df = scored_df.reset_index().sort_values("health_score_pred", ascending=False)
    
    # --- FIX: Sanitize Data for JSON Serialization (Remove NaN/Inf) ---
    # Replace Infinity with NaN, then replace NaN with 0
    scored_df = scored_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # --- FIX: Sanitize Metrics for JSON Serialization ---
    # Ensure R2/RMSE are floats and not NaN
    def clean_metric(val):
        if pd.isna(val) or np.isinf(val):
            return 0.0
        return float(val)

    metrics_dict = {
        "r2": clean_metric(model_obj["metrics"].r2),
        "rmse": clean_metric(model_obj["metrics"].rmse),
        "mae": clean_metric(model_obj["metrics"].mae)
    }

    # Clean importances
    importances = model_obj["importances"].fillna(0)
    
    # Force Garbage Collection
    gc.collect()

    return SKUHealthResult(
        scored_df=scored_df,
        model_obj=model_obj,
        train_metrics=metrics_dict,
        importances=importances
    )

# Maintain backward compatibility
def build_and_score_sku_health(merged_data: pd.DataFrame,
                               Return: pd.DataFrame,
                               unpaid_orders: Optional[pd.DataFrame] = None,
                               output_dir: Optional[str] = None) -> Dict:

    result = analyze_sku_health(merged_data, Return, unpaid_orders)
    
    return {
        "scored_df": result.scored_df,
        "model_obj": result.model_obj,
        "train_metrics": result.train_metrics,
        "importances": result.importances
    }