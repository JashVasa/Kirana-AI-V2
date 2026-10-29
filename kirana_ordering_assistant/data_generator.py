"""Synthetic Data Generator for AI-Powered Kirana Ordering Assistant.
Generates 300 Kirana stores and 180 days of simulated order history across 10 FMCG SKUs.
Grounds realistic store-level replenishment cycles for the Random Forest reorder model.
"""
import json
import random
import datetime
import os
import pandas as pd

random.seed(42)

AREAS = [
    ("Kalyan West", "Kalyan"),
    ("Kalyan East", "Kalyan"),
    ("Dombivli West", "Dombivli"),
    ("Dombivli East", "Dombivli"),
    ("Thane West", "Thane"),
    ("Naupada", "Thane"),
    ("Ghansoli", "Navi Mumbai"),
    ("Vashi", "Navi Mumbai"),
    ("Mulund West", "Mumbai"),
    ("Bhandup", "Mumbai")
]

STORE_TYPES = ["Small Kirana", "Medium Kirana", "Superette / Provision Store"]
OWNER_SURNAMES = ["Gupta", "Sharma", "Patel", "Yadav", "Jain", "Shah", "Choudhary", "Verma", "Pandey", "Mishra", "Thakur"]
OWNER_FIRSTNAMES = ["Ramesh", "Suresh", "Mahesh", "Rajesh", "Dinesh", "Mukesh", "Santosh", "Ashok", "Vijay", "Pravin"]

def load_catalog():
    catalog_path = os.path.join(os.path.dirname(__file__), "catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_stores(num_stores=300):
    stores = []
    for i in range(1, num_stores + 1):
        store_id = f"K{i:03d}"
        fname = random.choice(OWNER_FIRSTNAMES)
        lname = random.choice(OWNER_SURNAMES)
        owner_name = f"{fname} {lname}"
        area, city = random.choice(AREAS)
        store_type = random.choice(STORE_TYPES)
        store_name = f"{fname} Kirana & General Store"
        phone = f"98{random.randint(10000000, 99999999)}"
        
        # Define baseline store habit profile (SKUs ordered frequently)
        # Store K042 habit: regular buyer of Fortune Oil (SKU003) & Parle-G (SKU004)
        if store_id == "K042":
            core_skus = ["SKU003", "SKU004", "SKU005"]
            cycle_days = {"SKU003": 10, "SKU004": 7, "SKU005": 14}
        else:
            core_skus = random.sample(["SKU001", "SKU002", "SKU003", "SKU004", "SKU005", "SKU006", "SKU007", "SKU008", "SKU009", "SKU010"], k=random.randint(3, 6))
            cycle_days = {sku: random.choice([5, 7, 10, 14, 21]) for sku in core_skus}
            
        stores.append({
            "store_id": store_id,
            "store_name": store_name,
            "owner_name": owner_name,
            "phone": phone,
            "area": area,
            "city": city,
            "store_type": store_type,
            "core_skus": core_skus,
            "cycle_days": cycle_days
        })
    return stores

def generate_order_history(stores, catalog, days=180):
    catalog_by_sku = {p["sku"]: p for p in catalog}
    start_date = datetime.date(2026, 5, 1)
    
    orders = []
    order_counter = 10001
    
    # Track last order date for each store & sku
    store_last_order = {s["store_id"]: {sku["sku"]: None for sku in catalog} for s in stores}
    
    for day_offset in range(days):
        current_date = start_date + datetime.timedelta(days=day_offset)
        
        # On any given day, a subset of stores place an order
        active_stores = [s for s in stores if random.random() < 0.22]
        
        for store in active_stores:
            store_id = store["store_id"]
            order_id = f"ORD-{order_counter}"
            order_counter += 1
            
            # Decide which SKUs are ordered today
            ordered_skus = []
            for sku in store["core_skus"]:
                last_dt = store_last_order[store_id][sku]
                target_cycle = store["cycle_days"].get(sku, 10)
                if last_dt is None:
                    if random.random() < 0.4:
                        ordered_skus.append(sku)
                else:
                    days_since = (current_date - last_dt).days
                    # Probability of ordering increases as days_since approaches cycle
                    prob = min(0.9, max(0.05, (days_since / target_cycle) * 0.6))
                    if random.random() < prob:
                        ordered_skus.append(sku)
                        
            # Occasionally add a non-core SKU (impulse or promotional stock)
            if random.random() < 0.15:
                non_core = [sku for sku in catalog_by_sku.keys() if sku not in ordered_skus]
                if non_core:
                    ordered_skus.append(random.choice(non_core))
                    
            # For Store K042 in the last 14 days, deliberately do NOT order SKU003 (Fortune Oil)
            # so that it creates the exact reorder nudge scenario in the demo!
            if store_id == "K042" and day_offset > (days - 14):
                ordered_skus = [s for s in ordered_skus if s != "SKU003"]
                
            if not ordered_skus:
                continue
                
            for sku in ordered_skus:
                prod = catalog_by_sku[sku]
                qty = random.randint(1, 4)
                line_total = qty * prod["unit_price"]
                orders.append({
                    "order_id": order_id,
                    "store_id": store_id,
                    "order_date": current_date.isoformat(),
                    "sku": sku,
                    "product_name": prod["name"],
                    "category": prod["category"],
                    "quantity": qty,
                    "unit": prod["primary_unit"],
                    "unit_price": prod["unit_price"],
                    "line_total": line_total
                })
                store_last_order[store_id][sku] = current_date
                
    return orders

def main():
    catalog = load_catalog()
    base_dir = os.path.dirname(__file__)
    stores = generate_stores(300)
    
    with open(os.path.join(base_dir, "stores.json"), "w", encoding="utf-8") as f:
        json.dump(stores, f, indent=2)
    print(f"Generated {len(stores)} kirana stores -> stores.json")
    
    orders = generate_order_history(stores, catalog, 180)
    df_orders = pd.DataFrame(orders)
    csv_path = os.path.join(base_dir, "order_history.csv")
    df_orders.to_csv(csv_path, index=False)
    print(f"Generated {len(df_orders)} historical order line records -> order_history.csv")

if __name__ == "__main__":
    main()
