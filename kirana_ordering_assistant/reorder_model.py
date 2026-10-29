"""Predictive Reorder Model for AI-Powered Kirana Ordering Assistant.
Trains a Random Forest Classifier on historical kirana order history (180 days).
Predicts whether a store is due to reorder a specific SKU in the next 7 days.
Compares against a standard Frequency Baseline and generates contextual Hinglish nudges.
"""
import os
import datetime
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "reorder_rf_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
CATALOG_PATH = os.path.join(BASE_DIR, "catalog.json")
STORES_PATH = os.path.join(BASE_DIR, "stores.json")
ORDERS_PATH = os.path.join(BASE_DIR, "order_history.csv")

def load_data():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    with open(STORES_PATH, "r", encoding="utf-8") as f:
        stores = json.load(f)
    df_orders = pd.read_csv(ORDERS_PATH)
    df_orders["order_date"] = pd.to_datetime(df_orders["order_date"])
    return catalog, stores, df_orders

def build_dataset(df_orders, catalog, stores):
    """Builds (Store, SKU, Observation Date) feature table for 7-day reorder prediction."""
    min_date = df_orders["order_date"].min()
    max_date = df_orders["order_date"].max()
    
    # We choose observation cutoff dates every 7 days from day 45 to day 170
    dates = pd.date_range(min_date + pd.Timedelta(days=45), max_date - pd.Timedelta(days=8), freq="7D")
    
    skus = [p["sku"] for p in catalog]
    sku_popularity = df_orders["sku"].value_counts(normalize=True).to_dict()
    
    records = []
    
    for obs_date in dates:
        future_window = obs_date + pd.Timedelta(days=7)
        past_orders = df_orders[df_orders["order_date"] <= obs_date]
        future_orders = df_orders[(df_orders["order_date"] > obs_date) & (df_orders["order_date"] <= future_window)]
        
        # Build store-level history
        store_orders_grouped = past_orders.groupby(["store_id", "sku"])
        
        for store in stores:
            store_id = store["store_id"]
            store_past = past_orders[past_orders["store_id"] == store_id]
            total_past_store_orders = len(store_past)
            if total_past_store_orders == 0:
                continue
                
            for sku in skus:
                sku_past = store_past[store_past["sku"] == sku]
                if len(sku_past) == 0:
                    days_since_last = 90.0
                    avg_interval = 30.0
                    orders_last_30d = 0
                else:
                    last_order_dt = sku_past["order_date"].max()
                    days_since_last = (obs_date - last_order_dt).days
                    dates_sorted = sku_past["order_date"].sort_values().drop_duplicates()
                    if len(dates_sorted) > 1:
                        avg_interval = (dates_sorted.diff().dt.days.dropna()).mean()
                    else:
                        avg_interval = store["cycle_days"].get(sku, 14.0)
                    orders_last_30d = len(sku_past[sku_past["order_date"] >= (obs_date - pd.Timedelta(days=30))])
                    
                ratio_cycle = days_since_last / max(avg_interval, 1.0)
                
                # Ground truth target: did store order this SKU in next 7 days?
                will_reorder = 1 if len(future_orders[(future_orders["store_id"] == store_id) & (future_orders["sku"] == sku)]) > 0 else 0
                
                records.append({
                    "store_id": store_id,
                    "sku": sku,
                    "obs_date": obs_date,
                    "days_since_last": days_since_last,
                    "avg_interval": avg_interval,
                    "ratio_cycle": ratio_cycle,
                    "orders_last_30d": orders_last_30d,
                    "total_store_orders": total_past_store_orders,
                    "sku_popularity": sku_popularity.get(sku, 0.1),
                    "is_core": 1 if sku in store["core_skus"] else 0,
                    "target": will_reorder
                })
                
    return pd.DataFrame(records)

def train_and_evaluate():
    print("Loading data...")
    catalog, stores, df_orders = load_data()
    print("Building feature dataset for predictive reorder model...")
    df = build_dataset(df_orders, catalog, stores)
    
    # Chronological Split (Train before cutoff, Test after cutoff) as described in report
    cutoff = df["obs_date"].quantile(0.75)
    train_df = df[df["obs_date"] <= cutoff]
    test_df = df[df["obs_date"] > cutoff]
    
    features = ["days_since_last", "avg_interval", "ratio_cycle", "orders_last_30d", "total_store_orders", "sku_popularity", "is_core"]
    
    X_train = train_df[features]
    y_train = train_df["target"]
    X_test = test_df[features]
    y_test = test_df["target"]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    # Baseline: Rule-based frequency baseline (predict 1 if days_since >= 0.9 * avg_interval)
    y_pred_base = (test_df["ratio_cycle"] >= 0.9).astype(int)
    
    metrics = {
        "random_forest": {
            "accuracy": round(accuracy_score(y_test, y_pred_rf) * 100, 1),
            "precision": round(precision_score(y_test, y_pred_rf) * 100, 1),
            "recall": round(recall_score(y_test, y_pred_rf) * 100, 1),
            "f1": round(f1_score(y_test, y_pred_rf) * 100, 1),
            "cm": confusion_matrix(y_test, y_pred_rf).tolist()
        },
        "baseline": {
            "accuracy": round(accuracy_score(y_test, y_pred_base) * 100, 1),
            "precision": round(precision_score(y_test, y_pred_base) * 100, 1),
            "recall": round(recall_score(y_test, y_pred_base) * 100, 1),
            "f1": round(f1_score(y_test, y_pred_base) * 100, 1),
            "cm": confusion_matrix(y_test, y_pred_base).tolist()
        },
        "training_metadata": {
            "num_stores": len(stores),
            "num_orders": len(df_orders),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features_used": features,
            "feature_importances": {feat: round(imp, 4) for feat, imp in zip(features, rf.feature_importances_)}
        }
    }
    
    # Save model and metrics
    joblib.dump(rf, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print("\nModel Training Complete! Evaluation Summary:")
    print("-------------------------------------------------------------------")
    print(f"Random Forest   -> Acc: {metrics['random_forest']['accuracy']}%, Precision: {metrics['random_forest']['precision']}%, Recall: {metrics['random_forest']['recall']}%, F1: {metrics['random_forest']['f1']}%")
    print(f"Heuristic Base  -> Acc: {metrics['baseline']['accuracy']}%, Precision: {metrics['baseline']['precision']}%, Recall: {metrics['baseline']['recall']}%, F1: {metrics['baseline']['f1']}%")
    print(f"Feature Importances: {metrics['training_metadata']['feature_importances']}")
    return rf, metrics

def get_reorder_nudges_for_store(store_id: str, current_ordered_skus: list) -> list:
    """Evaluates whether any missing catalog item is overdue for replenishment for this store."""
    if not os.path.exists(ORDERS_PATH):
        return []
    
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    catalog_by_sku = {p["sku"]: p for p in catalog}
    
    with open(STORES_PATH, "r", encoding="utf-8") as f:
        stores = json.load(f)
    store_meta = next((s for s in stores if s["store_id"] == store_id), None)
    if not store_meta:
        return []
        
    df_orders = pd.read_csv(ORDERS_PATH)
    df_orders["order_date"] = pd.to_datetime(df_orders["order_date"])
    max_date = df_orders["order_date"].max()
    
    store_past = df_orders[df_orders["store_id"] == store_id]
    total_store_orders = len(store_past)
    sku_pop = df_orders["sku"].value_counts(normalize=True).to_dict()
    
    nudges = []
    
    # Check each SKU not present in the current order
    for sku, prod in catalog_by_sku.items():
        if sku in current_ordered_skus:
            continue
            
        sku_past = store_past[store_past["sku"] == sku]
        if len(sku_past) == 0:
            continue
            
        last_dt = sku_past["order_date"].max()
        days_since = (max_date - last_dt).days
        dates_sorted = sku_past["order_date"].sort_values().drop_duplicates()
        if len(dates_sorted) > 1:
            avg_interval = float((dates_sorted.diff().dt.days.dropna()).mean())
        else:
            avg_interval = float(store_meta["cycle_days"].get(sku, 14.0))
            
        ratio = days_since / max(avg_interval, 1.0)
        
        # High likelihood if days_since >= 85% of regular cycle and item is core
        is_core = sku in store_meta["core_skus"]
        if (ratio >= 0.85 and is_core) or (days_since >= 10 and is_core):
            urgency = "High" if ratio >= 1.2 else "Medium"
            
            # Formulate culturally natural, polite Hinglish reorder nudge
            friendly_name = prod["name"].split()[0] + " " + prod["name"].split()[1] if len(prod["name"].split()) > 1 else prod["name"]
            suggested_qty = int(sku_past["quantity"].median()) or 1
            unit = prod["primary_unit"]
            price = suggested_qty * prod["unit_price"]
            
            nudge_msg = f"Aapne is baar {friendly_name} order nahi kiya. Pichle order ko {days_since} din ho gaye hain. {suggested_qty} {unit} bhej dein?"
            
            nudges.append({
                "sku": sku,
                "product_name": prod["name"],
                "days_since_last": int(days_since),
                "avg_interval": round(avg_interval, 1),
                "ratio_cycle": round(ratio, 2),
                "suggested_qty": suggested_qty,
                "unit": unit,
                "unit_price": prod["unit_price"],
                "line_total": price,
                "urgency": urgency,
                "nudge_text": nudge_msg
            })
            
    nudges.sort(key=lambda x: x["ratio_cycle"], reverse=True)
    return nudges

if __name__ == "__main__":
    train_and_evaluate()
