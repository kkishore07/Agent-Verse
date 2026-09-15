import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def initialize_customers_and_transactions(workspace_dir: str):
    """
    Initializes a rich customer registry and transaction ledger for a workspace
    if they do not already exist or if they are empty.
    """
    if not workspace_dir:
        return
        
    customers_path = os.path.join(workspace_dir, "customers.json")
    tx_path = os.path.join(workspace_dir, "transactions.csv")
    inv_path = os.path.join(workspace_dir, "inventory.csv")
    
    # 1. Define standard mock customers
    default_customers = [
        {"id": "C101", "name": "Aarav Sharma", "email": "aarav.sharma@gmail.com", "phone": "+919876543210"},
        {"id": "C102", "name": "Deepika Patel", "email": "deepika.patel@gmail.com", "phone": "+919876543211"},
        {"id": "C103", "name": "Priya Nair", "email": "priya.nair@gmail.com", "phone": "+919876543212"},
        {"id": "C104", "name": "Rahul Verma", "email": "rahul.verma@gmail.com", "phone": "+919876543213"},
        {"id": "C105", "name": "Vikram Singh", "email": "vikram.singh@gmail.com", "phone": "+919876543214"},
        {"id": "C106", "name": "Ananya Rao", "email": "ananya.rao@gmail.com", "phone": "+919876543215"},
        {"id": "C107", "name": "Amit Gupta", "email": "amit.gupta@gmail.com", "phone": "+919876543216"},
        {"id": "C108", "name": "Sneha Reddy", "email": "sneha.reddy@gmail.com", "phone": "+919876543217"},
        {"id": "C109", "name": "Rohan Das", "email": "rohan.das@gmail.com", "phone": "+919876543218"},
        {"id": "C110", "name": "Meera Krishnan", "email": "meera.krishnan@gmail.com", "phone": "+919876543219"}
    ]
    
    # Write customers registry if missing
    if not os.path.exists(customers_path):
        with open(customers_path, "w", encoding="utf-8") as f:
            json.dump(default_customers, f, indent=4)
            
    # Load inventory to know products
    inventory_items = []
    if os.path.exists(inv_path):
        try:
            df_inv = pd.read_csv(inv_path)
            if not df_inv.empty:
                inventory_items = df_inv.to_dict('records')
        except Exception:
            pass
            
    if not inventory_items:
        # Fallback inventory items
        inventory_items = [
            {"ProductID": "P101", "ProductName": "Premium Basmati Rice 5kg", "Category": "Grains", "UnitPrice": 350.0, "RetailPrice": 450.0, "Supplier": "Sri Balaji Traders"},
            {"ProductID": "P102", "ProductName": "Gold Winner Sunflower Oil 1L", "Category": "Oils", "UnitPrice": 110.0, "RetailPrice": 140.0, "Supplier": "Vignesh Wholesalers"},
            {"ProductID": "P103", "ProductName": "Tata Salt 1kg", "Category": "Condiments", "UnitPrice": 20.0, "RetailPrice": 28.0, "Supplier": "Tirupur Distributors"},
            {"ProductID": "P104", "ProductName": "Aashirvaad Shudh Chakki Atta 5kg", "Category": "Grains", "UnitPrice": 220.0, "RetailPrice": 280.0, "Supplier": "Sri Balaji Traders"},
            {"ProductID": "P105", "ProductName": "Toor Dal Premium 1kg", "Category": "Pulses", "UnitPrice": 140.0, "RetailPrice": 180.0, "Supplier": "Raja Pulses"},
            {"ProductID": "P106", "ProductName": "Britannia Marie Gold Biscuits", "Category": "Snacks", "UnitPrice": 12.0, "RetailPrice": 15.0, "Supplier": "Tirupur Distributors"},
            {"ProductID": "P107", "ProductName": "Brooke Bond Red Label Tea 250g", "Category": "Beverages", "UnitPrice": 95.0, "RetailPrice": 120.0, "Supplier": "Vignesh Wholesalers"},
            {"ProductID": "P108", "ProductName": "Sunsilk Black Shampoo 180ml", "Category": "Personal Care", "UnitPrice": 115.0, "RetailPrice": 145.0, "Supplier": "Vignesh Wholesalers"}
        ]
        # Save inventory as well if empty or missing
        if not os.path.exists(inv_path) or (os.path.exists(inv_path) and os.path.getsize(inv_path) < 100):
            df_inv = pd.DataFrame([
                {"ProductID": item["ProductID"], "ProductName": item["ProductName"], "Category": item["Category"], 
                 "StockLevel": 45, "ReorderLevel": 15, "UnitPrice": item["UnitPrice"], "RetailPrice": item["RetailPrice"], 
                 "Supplier": item["Supplier"]} for item in inventory_items
            ])
            df_inv.to_csv(inv_path, index=False)

    # Check transactions
    has_rich_tx = False
    if os.path.exists(tx_path):
        try:
            df_tx = pd.read_csv(tx_path)
            if len(df_tx) > 20 and 'CustomerID' in df_tx.columns and df_tx['CustomerID'].notna().sum() > 10:
                has_rich_tx = True
        except Exception:
            pass
            
    if not has_rich_tx:
        # Generate rich historical sales mapping to our customers
        today = datetime.now()
        np.random.seed(42)
        transactions = []
        
        # Design specific timelines for customer segments
        purchase_timelines = {
            "C101": [today - timedelta(days=d) for d in [3, 10, 18, 25, 34, 45, 60, 80, 100]], # VIP
            "C102": [today - timedelta(days=d) for d in [5, 12, 22, 35, 50, 70, 90]], # VIP
            "C103": [today - timedelta(days=d) for d in [10, 20, 42, 65]], # Regular
            "C104": [today - timedelta(days=d) for d in [75, 95]], # Inactive
            "C105": [today - timedelta(days=d) for d in [35, 70, 80]], # Decreased frequency (gap: 10d -> 35d)
            "C106": [today - timedelta(days=4)], # New
            "C107": [today - timedelta(days=95)], # Inactive
            "C108": [today - timedelta(days=d) for d in [12, 24, 38]], # Regular
            "C109": [today - timedelta(days=25)], # Occasional
            "C110": [today - timedelta(days=d) for d in [8, 16, 28, 48]] # Regular
        }
        
        # Products purchased distribution to capture commonly bought together (Rice & Dal, Tea & Biscuits)
        inv_by_id = {str(item["ProductID"]): item for item in inventory_items}
        available_ids = list(inv_by_id.keys())
        if not available_ids:
            available_ids = ["P101"]
            
        def get_best_pref(preferred_list, cid_seed):
            existing = [p for p in preferred_list if p in available_ids]
            if existing:
                return existing
            import random
            random.seed(hash(cid_seed))
            return random.sample(available_ids, min(len(available_ids), 3))

        customer_item_preferences = {
            "C101": get_best_pref(["P101", "P105", "P103"], "C101"),
            "C102": get_best_pref(["P101", "P105", "P104"], "C102"),
            "C103": get_best_pref(["P102", "P106", "P107"], "C103"),
            "C104": get_best_pref(["P101", "P103"], "C104"),
            "C105": get_best_pref(["P102", "P107"], "C105"),
            "C106": get_best_pref(["P104", "P105"], "C106"),
            "C107": get_best_pref(["P101"], "C107"),
            "C108": get_best_pref(["P107", "P106", "P108"], "C108"),
            "C109": get_best_pref(["P104"], "C109"),
            "C110": get_best_pref(["P101", "P105", "P102"], "C110")
        }
        
        txn_counter = 5000
        
        # Helper to find customer details
        cust_map = {c["id"]: c["name"] for c in default_customers}
        
        for cid, dates in purchase_timelines.items():
            cname = cust_map[cid]
            prefs = customer_item_preferences[cid]
            
            for dt in dates:
                txn_counter += 1
                txn_id = f"TXN{txn_counter}"
                
                # Frequently bought together logic: buy 1 or more items in the transaction
                # Pick a primary item from preferences
                main_item_id = np.random.choice(prefs)
                main_item = next((x for x in inventory_items if str(x["ProductID"]) == str(main_item_id)), inventory_items[0])
                qty = int(np.random.choice([1, 2], p=[0.7, 0.3]))
                price = main_item["RetailPrice"]
                
                transactions.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "TransactionID": txn_id,
                    "CustomerID": cid,
                    "CustomerName": cname,
                    "ProductID": main_item_id,
                    "ProductName": main_item["ProductName"],
                    "Category": main_item["Category"],
                    "Quantity": qty,
                    "Price": price,
                    "Type": "Sale",
                    "Amount": qty * price,
                    "PaymentMode": np.random.choice(["UPI", "Cash", "Card"], p=[0.7, 0.2, 0.1])
                })
                
                # Co-occurrence injection: if Rice, 60% chance to buy Dal. If Tea, 75% chance to buy Biscuits.
                if main_item_id == "P101" and np.random.rand() < 0.7:
                    # Rice -> Dal (P105)
                    dal = next((x for x in inventory_items if x["ProductID"] == "P105"), None)
                    if dal:
                        transactions.append({
                            "Date": dt.strftime("%Y-%m-%d"),
                            "TransactionID": txn_id,
                            "CustomerID": cid,
                            "CustomerName": cname,
                            "ProductID": "P105",
                            "ProductName": dal["ProductName"],
                            "Category": dal["Category"],
                            "Quantity": 1,
                            "Price": dal["RetailPrice"],
                            "Type": "Sale",
                            "Amount": dal["RetailPrice"],
                            "PaymentMode": transactions[-1]["PaymentMode"]
                        })
                elif main_item_id == "P107" and np.random.rand() < 0.75:
                    # Tea -> Biscuits (P106)
                    biscuits = next((x for x in inventory_items if x["ProductID"] == "P106"), None)
                    if biscuits:
                        transactions.append({
                            "Date": dt.strftime("%Y-%m-%d"),
                            "TransactionID": txn_id,
                            "CustomerID": cid,
                            "CustomerName": cname,
                            "ProductID": "P106",
                            "ProductName": biscuits["ProductName"],
                            "Category": biscuits["Category"],
                            "Quantity": 2,
                            "Price": biscuits["RetailPrice"],
                            "Type": "Sale",
                            "Amount": 2 * biscuits["RetailPrice"],
                            "PaymentMode": transactions[-1]["PaymentMode"]
                        })
                    
        # Add a couple of supplier restock operations (Expenses) to keep transactions realistic
        transactions.append({
            "Date": (today - timedelta(days=20)).strftime("%Y-%m-%d"),
            "TransactionID": "PO_RESTOCK_C1",
            "CustomerID": "",
            "CustomerName": "",
            "ProductID": "P101",
            "ProductName": "Restock - Premium Basmati Rice 5kg",
            "Category": "Grains",
            "Quantity": 20,
            "Price": 350.0,
            "Type": "Expense",
            "Amount": 7000.0,
            "PaymentMode": "UPI"
        })
        transactions.append({
            "Date": (today - timedelta(days=15)).strftime("%Y-%m-%d"),
            "TransactionID": "PO_RESTOCK_C2",
            "CustomerID": "",
            "CustomerName": "",
            "ProductID": "P102",
            "ProductName": "Restock - Gold Winner Sunflower Oil 1L",
            "Category": "Oils",
            "Quantity": 30,
            "Price": 110.0,
            "Type": "Expense",
            "Amount": 3300.0,
            "PaymentMode": "Cash"
        })
        
        # Save to csv
        df_tx_new = pd.DataFrame(transactions)
        df_tx_new = df_tx_new.sort_values("Date")
        df_tx_new.to_csv(tx_path, index=False)

def get_customer_insights(workspace_dir: str, config_inactive_days: int = 60) -> dict:
    """
    Performs dynamic customer cohort analysis and association rule calculations.
    Returns:
        dict containing metrics, top customers list, segmentation profile,
        purchase trends, recommendations, and frequently bought together items.
    """
    # Initialize workspace details if not present
    initialize_customers_and_transactions(workspace_dir)
    
    customers_path = os.path.join(workspace_dir, "customers.json") if workspace_dir else ""
    tx_path = os.path.join(workspace_dir, "transactions.csv") if workspace_dir else ""
    
    # Fallback to load default datasets if files don't exist
    if not workspace_dir or not os.path.exists(customers_path) or not os.path.exists(tx_path):
        return {
            "metrics": {
                "total_customers": 0,
                "active_customers": 0,
                "inactive_customers": 0,
                "repeat_customer_rate": 0.0,
                "average_order_value": 0.0,
                "customer_lifetime_value": 0.0
            },
            "customers": [],
            "segments": {"VIP": 0, "Regular": 0, "Occasional": 0, "New": 0, "Inactive": 0},
            "trends": [],
            "frequently_bought_together": [],
            "recommendations": [],
            "is_empty": True
        }
        
    customers_list = []
    if os.path.exists(customers_path):
        try:
            with open(customers_path, "r", encoding="utf-8") as f:
                customers_list = json.load(f)
        except Exception:
            customers_list = []

    # Merge any customers stored in MongoDB for this workspace
    workspace_id = os.path.basename(workspace_dir) if workspace_dir else None
    if workspace_id:
        try:
            from database import db
            mongo_custs = list(db.customers.find({"workspace_id": workspace_id}, {"_id": 0}))
            for mc in mongo_custs:
                if not any(
                    (mc.get("email") and str(c.get("email", "")).lower() == str(mc["email"]).lower()) or
                    (mc.get("phone") and str(c.get("phone", "")).strip() == str(mc["phone"]).strip()) or
                    (mc.get("id") and str(c.get("id", "")) == str(mc["id"]))
                    for c in customers_list
                ):
                    customers_list.append(mc)
        except Exception as me:
            pass
        
    tx_df = pd.read_csv(tx_path)
    
    # Focus only on Sales
    sales_df = tx_df[(tx_df['Type'] == 'Sale') & (tx_df['CustomerID'].notna())]
    
    if sales_df.empty:
        # Construct empty result maps
        return {
            "metrics": {
                "total_customers": len(customers_list),
                "active_customers": 0,
                "inactive_customers": len(customers_list),
                "repeat_customer_rate": 0.0,
                "average_order_value": 0.0,
                "customer_lifetime_value": 0.0
            },
            "customers": [
                {**c, "spending": 0.0, "visits": 0, "last_purchase": "N/A", 
                 "payment_method": "N/A", "pref_products": [], "segment": "Inactive"} 
                for c in customers_list
            ],
            "segments": {"VIP": 0, "Regular": 0, "Occasional": 0, "New": 0, "Inactive": len(customers_list)},
            "trends": [],
            "frequently_bought_together": [],
            "recommendations": [],
            "is_empty": False
        }
        
    # Cast/Parse columns safely
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    sales_df['CustomerID'] = sales_df['CustomerID'].astype(str)
    sales_df['Amount'] = sales_df['Amount'].astype(float)
    sales_df['Quantity'] = sales_df['Quantity'].astype(int)
    
    today = datetime.now()
    
    customer_profiles = []
    segment_counts = {"VIP": 0, "Regular": 0, "Occasional": 0, "New": 0, "Inactive": 0}
    
    # Loop over registered customers to compile profiles
    for cust in customers_list:
        cid = str(cust["id"])
        cname = cust["name"]
        
        c_sales = sales_df[sales_df['CustomerID'] == cid]
        
        if c_sales.empty:
            profile = {
                "id": cid,
                "name": cname,
                "email": cust.get("email", ""),
                "phone": cust.get("phone", ""),
                "spending": 0.0,
                "visits": 0,
                "last_purchase": "N/A",
                "days_since_last": 999,
                "payment_method": "N/A",
                "pref_products": [],
                "purchase_history": [],
                "segment": "New"
            }
            segment_counts["New"] += 1
            customer_profiles.append(profile)
            continue
            
        # Calculate spending metrics
        total_spent = float(c_sales['Amount'].sum())
        
        # Visits: Number of unique purchase dates
        unique_dates = c_sales['Date'].dt.date.unique()
        visits = len(unique_dates)
        
        # Last purchase date
        last_purch_dt = c_sales['Date'].max()
        last_purch_str = last_purch_dt.strftime("%Y-%m-%d")
        days_since = (today - last_purch_dt).days
        
        # Preferred payment mode
        pref_pay = str(c_sales.groupby('PaymentMode')['Amount'].sum().idxmax()) if 'PaymentMode' in c_sales.columns and not c_sales.empty else "UPI"
        
        # Frequently purchased products (limit to top 3)
        top_prods = c_sales.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(3).index.tolist()
        
        # Purchase history details for timelines
        history = []
        for _, row in c_sales.sort_values('Date', ascending=False).iterrows():
            history.append({
                "date": row['Date'].strftime("%Y-%m-%d"),
                "txn_id": row['TransactionID'],
                "product": row['ProductName'],
                "qty": int(row['Quantity']),
                "amount": float(row['Amount']),
                "payment": row['PaymentMode']
            })
            
        # Segment definitions
        first_purch_dt = c_sales['Date'].min()
        days_since_first = (today - first_purch_dt).days
        
        if days_since > config_inactive_days:
            seg = "Inactive"
        elif total_spent >= 5000.0 and visits >= 5:
            seg = "VIP"
        elif visits >= 3:
            seg = "Regular"
        elif days_since_first <= 15 and visits < 3:
            seg = "New"
        else:
            seg = "Occasional"
            
        segment_counts[seg] += 1
        
        customer_profiles.append({
            "id": cid,
            "name": cname,
            "email": cust["email"],
            "phone": cust["phone"],
            "spending": round(total_spent, 2),
            "visits": visits,
            "last_purchase": last_purch_str,
            "days_since_last": days_since,
            "payment_method": pref_pay,
            "pref_products": top_prods,
            "purchase_history": history,
            "segment": seg
        })
        
    # Calculate Overall Cohort Metrics
    total_customers = len(customer_profiles)
    active_customers = sum(1 for c in customer_profiles if c["segment"] != "Inactive")
    inactive_customers = total_customers - active_customers
    
    repeat_customers = sum(1 for c in customer_profiles if c["visits"] >= 2)
    repeat_rate = round((repeat_customers / total_customers * 100.0), 1) if total_customers > 0 else 0.0
    
    # Average Order Value (Group by TransactionID)
    order_sums = sales_df.groupby('TransactionID')['Amount'].sum()
    aov = float(order_sums.mean()) if not order_sums.empty else 0.0
    
    # Customer Lifetime Value = total sales / total customers
    total_revenue = float(sales_df['Amount'].sum())
    clv = float(total_revenue / total_customers) if total_customers > 0 else 0.0
    
    metrics = {
        "total_customers": total_customers,
        "active_customers": active_customers,
        "inactive_customers": inactive_customers,
        "repeat_customer_rate": repeat_rate,
        "average_order_value": round(aov, 2),
        "customer_lifetime_value": round(clv, 2),
        "total_revenue": round(total_revenue, 2)
    }
    
    # Top Customers List
    sorted_profiles = sorted(customer_profiles, key=lambda x: x["spending"], reverse=True)
    
    # Purchase Trends: aggregate sales and active buyer counts month-by-month (past 6 months)
    sales_df['YearMonth'] = sales_df['Date'].dt.to_period('M')
    trends = []
    
    # Sort year-months chronologically
    sorted_months = sorted(sales_df['YearMonth'].unique())
    for ym in sorted_months[-6:]: # last 6 months
        group = sales_df[sales_df['YearMonth'] == ym]
        trends.append({
            "month": ym.strftime("%B %Y"),
            "revenue": round(float(group['Amount'].sum()), 2),
            "customer_count": int(group['CustomerID'].nunique()),
            "order_count": int(group['TransactionID'].nunique())
        })
        
    # Frequently Bought Together (Association Analysis)
    # Group by (CustomerID, Date) to identify co-purchases during the same visit
    basket_groups = sales_df.groupby(['CustomerID', sales_df['Date'].dt.date])
    baskets = []
    for _, grp in basket_groups:
        prod_names = grp['ProductName'].dropna().unique().tolist()
        if len(prod_names) > 1:
            baskets.append(prod_names)
            
    # Count pairs of products
    pair_counts = {}
    for basket in baskets:
        # Generate all distinct pairs
        basket_sorted = sorted(basket)
        for i in range(len(basket_sorted)):
            for j in range(i + 1, len(basket_sorted)):
                pair = (basket_sorted[i], basket_sorted[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                
    fbt_list = []
    # Sort pairs by co-occurrence count
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    for pair, count in sorted_pairs[:5]: # top 5 pairs
        # Calculate support: co-occurrence / total baskets (or total sales invoices)
        support = round((count / len(order_sums) * 100.0), 1) if not order_sums.empty else 0.0
        fbt_list.append({
            "product_a": pair[0],
            "product_b": pair[1],
            "co_occurrence": count,
            "support_percentage": support
        })
        
    # Generate Dynamic Actionable Recommendations
    recommendations = []
    
    # Recommendation 1: Re-engagement for Inactive Customers
    inactive_members = [c for c in customer_profiles if c["segment"] == "Inactive"]
    for ic in inactive_members[:3]: # top 3 inactive
        recommendations.append({
            "type": "re_engagement",
            "title": f"Re-engage {ic['name']}",
            "customer_id": ic["id"],
            "customer_name": ic["name"],
            "customer_phone": ic["phone"],
            "message": f"Customer hasn't purchased in {ic['days_since_last']} days (Last purchase: {ic['last_purchase']}). Re-engage them with a WhatsApp discount campaign.",
            "suggested_action": "WhatsApp Promo Code",
            "discount_offer": "10% off using code WELCOMEBACK10",
            "context": f"Inactive segment campaign for {ic['name']}. Phone: {ic['phone']}"
        })
        
    # Recommendation 2: Decreased Frequency Promotional Alert
    for cp in customer_profiles:
        # Check if customer has at least 3 historical transactions to establish frequency
        if cp["visits"] >= 3 and cp["segment"] != "Inactive":
            history_dates = sorted([datetime.strptime(h["date"], "%Y-%m-%d") for h in cp["purchase_history"]])
            # Calculate average gap days
            gaps = [(history_dates[i] - history_dates[i-1]).days for i in range(1, len(history_dates))]
            avg_gap = np.mean(gaps) if gaps else 15.0
            
            # Days since last purchase
            days_since = cp["days_since_last"]
            
            # If current idle time is 1.8x larger than average shopping gap, flag frequency drop
            if days_since > avg_gap * 1.8 and days_since > 7:
                pref_cat = "Grains"
                # Lookup primary category from profile
                c_sales = sales_df[sales_df['CustomerID'] == cp["id"]]
                if not c_sales.empty:
                    pref_cat = c_sales.groupby('Category')['Amount'].sum().idxmax()
                    
                recommendations.append({
                    "type": "retention",
                    "title": f"Frequency Drop Alert: {cp['name']}",
                    "customer_id": cp["id"],
                    "customer_name": cp["name"],
                    "customer_phone": cp["phone"],
                    "message": f"Typical purchase gap is {round(avg_gap, 1)} days, but they haven't bought in {days_since} days. Suggested customized discount on {pref_cat} to secure retention.",
                    "suggested_action": "Frequency retention coupon",
                    "discount_offer": f"15% off {pref_cat} products with code SPECIAL15",
                    "context": f"Retention campaign for {cp['name']}. Target Category: {pref_cat}"
                })
                
    # Recommendation 3: Cross-selling & Bundle Opportunities
    # Suggest bundle purchase if customer buys one item but not the frequently-associated item
    for cp in customer_profiles:
        for fbt in fbt_list:
            bought_a = any(fbt["product_a"] in h["product"] for h in cp["purchase_history"])
            bought_b = any(fbt["product_b"] in h["product"] for h in cp["purchase_history"])
            
            if bought_a and not bought_b:
                recommendations.append({
                    "type": "cross_sell",
                    "title": f"Cross-sell {fbt['product_b']} to {cp['name']}",
                    "customer_id": cp["id"],
                    "customer_name": cp["name"],
                    "customer_phone": cp["phone"],
                    "message": f"They buy {fbt['product_a']} frequently. Recommend bundling {fbt['product_b']} based on common customer checkouts.",
                    "suggested_action": "Bundle discount offer",
                    "discount_offer": f"Save 5% on adding {fbt['product_b']} to Rice checkout",
                    "context": f"Cross-sell offer for {cp['name']}. Product A: {fbt['product_a']} | Product B: {fbt['product_b']}"
                })
                break # Limit to 1 cross-sell card per customer to avoid clutter
                
    return {
        "metrics": metrics,
        "customers": sorted_profiles,
        "segments": segment_counts,
        "trends": trends,
        "frequently_bought_together": fbt_list,
        "recommendations": recommendations,
        "is_empty": False
    }
