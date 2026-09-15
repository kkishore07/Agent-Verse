import os
import json
import config
import pandas as pd
from agents.base_agent import call_llm
from agents.finance_agent import run_finance_agent
from agents.inventory_agent import run_inventory_agent
from agents.analytics_agent import run_analytics_agent
from agents.communication_agent import run_communication_agent
from agents.supplier_agent import run_supplier_agent
from agents.customer_agent import run_customer_agent
from agents.strategy_agent import generate_strategy


def get_customer_db_context(workspace_dir: str = None) -> str:
    """Loads customer statistics, top buyers, segments, and bundle co-occurrences from analytics engine."""
    try:
        from forecasting.customer_analytics import get_customer_insights
        insights = get_customer_insights(workspace_dir)
        if insights.get("is_empty", False):
            return "No customer database records or sale transaction history found."
            
        metrics = insights["metrics"]
        cust_lines = []
        for c in insights["customers"][:5]: # top 5
            cust_lines.append(f"  - Name: {c['name']} (ID: {c['id']}) | Spent: Rs. {c['spending']:.2f} | Visits: {c['visits']} | Last: {c['last_purchase']} | Segment: {c['segment']} | Preferred: {', '.join(c['pref_products'])}")
        cust_str = "\n".join(cust_lines)
        
        fbt_lines = []
        for f in insights["frequently_bought_together"][:3]:
            fbt_lines.append(f"  - {f['product_a']} + {f['product_b']} (co-occurred {f['co_occurrence']} times)")
        fbt_str = "\n".join(fbt_lines) if fbt_lines else "No strong product bundles detected."
        
        recs_lines = []
        for r in insights["recommendations"][:4]:
            recs_lines.append(f"  - Action: {r['title']} | Offer: {r['discount_offer']} | Reason: {r['message']}")
        recs_str = "\n".join(recs_lines) if recs_lines else "No campaign promotions recommended at this time."
        
        segments_str = ", ".join([f"{k}: {v}" for k, v in insights["segments"].items()])
        
        return (
            f"Active Live Customer Insights Database (computed dynamically):\n"
            f"- Total Customers: {metrics['total_customers']} | Active: {metrics['active_customers']} | Inactive: {metrics['inactive_customers']}\n"
            f"- Average Order Value (AOV): Rs. {metrics['average_order_value']:.2f}\n"
            f"- Customer Lifetime Value (CLV): Rs. {metrics['customer_lifetime_value']:.2f}\n"
            f"- Repeat Buyer Rate: {metrics['repeat_customer_rate']}%\n"
            f"- Segments Distribution: {segments_str}\n"
            f"- Top Customers Profile:\n{cust_str}\n"
            f"- Frequently Bought Together Bundles:\n{fbt_str}\n"
            f"- Recommended Customer Retention Campaigns:\n{recs_str}"
        )
    except Exception as e:
        return f"Warning: Could not fetch active customer records ({str(e)})"


def get_supplier_db_context(workspace_dir: str = None) -> str:
    """Loads supplier list from MongoDB for the given workspace."""
    try:
        if not workspace_dir:
            return "No supplier database records found."
        workspace_id = os.path.basename(workspace_dir)
        from database import db
        suppliers = list(db.suppliers.find({"workspace_id": workspace_id}, {"_id": 0}))
        if not suppliers:
            return "Supplier catalog is currently empty."
        lines = ["Active Live Supplier Database (includes unit prices, MOQ, lead times, and reliability):"]
        for s in suppliers:
            catalog_lines = [
                f"Product: {item['product_id']} | Price: Rs. {item['unit_price']} | MOQ: {item['min_order_qty']} | Lead Time: {item['lead_time_days']} days"
                for item in s.get("catalog", [])
            ]
            lines.append(
                f"- Supplier: {s['name']} (ID: {s['id']}) | Reliability: {s['reliability']}% | "
                f"Terms: {s['payment_terms']} | Catalog: [{' | '.join(catalog_lines)}]"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Warning: Could not fetch active supplier records ({str(e)})"

def get_inventory_db_context(workspace_dir: str = None) -> str:
    """Loads inventory stock status, reorder guidelines, and suppliers from the live CSV."""
    from forecasting.predictor import predict_inventory_exhaustion
    try:
        depletion_data = predict_inventory_exhaustion(workspace_dir)
        if not depletion_data:
            return "No inventory database items found."
        
        lines = ["Active Live Inventory Database (includes Stock Levels, safety limits, daily velocities, and suppliers):"]
        for item in depletion_data:
            lines.append(
                f"- ProductID: {item['ProductID']} | Name: {item['ProductName']} | Category: {item['Category']} | "
                f"CurrentStock: {item['CurrentStock']} units | ReorderLevel (Safety Limit): {item['ReorderLevel']} units | "
                f"Daily Sales Velocity: {item['DailyVelocity']} units/day | DaysRemaining: {item['DaysRemaining']} days | "
                f"Status: {item['Status']} | Recommended Restock: {item['ReorderRecommendation']} units | "
                f"Supplier: {item['Supplier']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Warning: Could not fetch active inventory records ({str(e)})"

def get_finance_db_context(workspace_dir: str = None) -> str:
    """Loads transaction sums, category splits, and spike expense items from transactions ledger."""
    try:
        from forecasting.predictor import TRANSACTIONS_PATH
        tx_path = os.path.join(workspace_dir, 'transactions.csv') if workspace_dir else TRANSACTIONS_PATH
        if not os.path.exists(tx_path):
            return "No financial transaction logs found."
        tx_df = pd.read_csv(tx_path)
        if tx_df.empty:
            return "Transactions database is currently empty."
            
        sales = tx_df[tx_df['Type'] == 'Sale']['Amount'].sum()
        expenses = tx_df[tx_df['Type'] == 'Expense']['Amount'].sum()
        profit = sales - expenses
        
        # Category Sales
        cat_sales = tx_df[tx_df['Type'] == 'Sale'].groupby('Category')['Amount'].sum().to_dict()
        cat_str = ", ".join([f"{cat}: Rs. {amt:,.2f}" for cat, amt in cat_sales.items()])
        
        # Payment mode split
        pm_sales = tx_df[tx_df['Type'] == 'Sale'].groupby('PaymentMode')['Amount'].sum().to_dict()
        pm_str = ", ".join([f"{mode}: Rs. {amt:,.2f}" for mode, amt in pm_sales.items()])
        
        # Top expenses
        top_exp = tx_df[tx_df['Type'] == 'Expense'].sort_values('Amount', ascending=False).head(5)
        top_exp_lines = []
        for _, row in top_exp.iterrows():
            top_exp_lines.append(f"  - {row['ProductName']} on {row['Date']}: Rs. {row['Amount']:,.2f} ({row['PaymentMode']})")
        top_exp_str = "\n".join(top_exp_lines)
        
        return (
            f"Active Live Financial Ledger (from transactions.csv):\n"
            f"- Total Revenue (Sales): Rs. {sales:,.2f}\n"
            f"- Total Expenses: Rs. {expenses:,.2f}\n"
            f"- Net Profit Margin: Rs. {profit:,.2f}\n"
            f"- Sales Payment Methods Split: {pm_str}\n"
            f"- Sales by Category: {cat_str}\n"
            f"- Top Expense Spikes (Recent Restocks/Bills):\n{top_exp_str}"
        )
    except Exception as e:
        return f"Warning: Could not fetch active ledger statistics ({str(e)})"

def get_analytics_db_context(query: str, workspace_dir: str = None) -> str:
    """Generates a 30-day forecast and regression data for the active question."""
    try:
        from forecasting.predictor import forecast_sales, predict_inventory_exhaustion
        # Try to identify a product reference in query
        prod_id = None
        q_lower = query.lower()
        for pid in ["P101", "P102", "P103", "P104", "P105", "P106", "P107", "P108"]:
            if pid.lower() in q_lower:
                prod_id = pid
                break
                
        forecast_res = forecast_sales(product_id=prod_id, days_ahead=30, workspace_dir=workspace_dir)
        growth = forecast_res.get("growth_rate", 8.52)
        total_forecast = forecast_res.get("total_forecasted_sales", 0.0)
        prod_label = forecast_res.get("product_name") or "Total Business"
        
        future_dates = forecast_res.get("forecast", {}).get("dates", [])
        future_sales = forecast_res.get("forecast", {}).get("sales", [])
        
        sample_lines = []
        for d, s in zip(future_dates[:5], future_sales[:5]):
            sample_lines.append(f"  - Date: {d} | Forecasted sales: Rs. {s:,.2f}")
        sample_forecast_str = "\n".join(sample_lines) if sample_lines else "No forecast points available."
        
        return (
            f"Machine Learning (Linear Regression) 30-Day Sales Forecast Details:\n"
            f"- Target Product Analysis: {prod_label}\n"
            f"- Projected Revenue over next 30 days: Rs. {total_forecast:,.2f}\n"
            f"- Growth rate predicted: {growth}%\n"
            f"- Sample forecast dates & amounts:\n{sample_forecast_str}"
        )
    except Exception as e:
        return f"Warning: Could not run forecasting analysis ({str(e)})"

COORDINATOR_SYSTEM_INSTRUCTION = (
    "You are the Master Coordinator of AegisAI, a business copilot for Indian SMEs. "
    "Your job is to read the user's query and decide which specialized agent should process it: "
    "- FINANCE: For questions about money, revenue, costs, bills, sales values, profits, UPI vs Cash. "
    "- INVENTORY: For stock quantities, supplier names, reordering lists, low stock warnings. "
    "- ANALYTICS: For sales forecasting, future trends, growth calculations, explaining charts. "
    "- COMMUNICATION: For drafting WhatsApp alerts, customer messages, or notification texts. "
    "- SUPPLIER: For purchase orders, supplier evaluation, price quotes, procurement, or reordering recommendations. "
    "- CUSTOMER: For customer databases, loyalty, repeat buyers, spending, segmentation, CLV, AOV, purchase history, and campaigns. "
    "- STRATEGY: For questions about business strategy, growth, marketing, promotions, pricing ideas, branding, or attracting more customers. "
    "\n"
    "You must return your response in this exact format:\n"
    "AGENT: [FINANCE/INVENTORY/ANALYTICS/COMMUNICATION/SUPPLIER/CUSTOMER/STRATEGY/GENERAL]\n"
    "REASONING: [Explain in one brief sentence why you chose this agent]\n"
    "THOUGHTS: [List any additional coordinator steps, e.g., 'Analyzing invoice context...']"
)

def coordinate_agents(query: str, context: str, provider: str = "gemini", workspace_dir: str = None) -> dict:
    """
    Coordinates agent routing and executes the selected specialized agent.
    Returns:
        dict: {
            "response": str (Final output),
            "agent": str (Selected agent name),
            "reasoning": str (Selected agent reasoning),
            "thoughts": str (Detailed thoughts)
        }
    """
    query_lower = query.lower()
    
    # 1. Determine Routing (using LLM or static keywords fallback)
    routing_response = ""
    api_key = config.get_groq_key() or config.get_gemini_key()
    
    if api_key:
        try:
            prompt = f"User Query: {query}\n\nContext snippet:\n{context[:300]}"
            routing_response = call_llm(COORDINATOR_SYSTEM_INSTRUCTION, prompt, temperature=0.1, provider=provider)
        except Exception as e:
            print(f"Coordinator LLM routing failed ({e}). Using keyword routing.")
            
    if not routing_response:
        # Fallback keyword routing
        if any(k in query_lower for k in ["revenue", "sales", "finance", "expense", "profit", "earn", "விற்பனை", "வருவாய்", "இலாபம்"]):
            routing_response = "AGENT: FINANCE\nREASONING: The query deals with financial performance and sales amounts.\nTHOUGHTS: Routing to Finance Agent for ledger analysis."
        elif any(k in query_lower for k in ["stock", "inventory", "reorder", "restock", "low", "பொருள்", "இருப்பு", "தேவை"]):
            routing_response = "AGENT: INVENTORY\nREASONING: The query asks about stock status, quantities, or suppliers.\nTHOUGHTS: Routing to Inventory Agent to check warehouse logs."
        elif any(k in query_lower for k in ["predict", "forecast", "future", "next month", "கணிப்பு", "அடுத்த மாதம்"]):
            routing_response = "AGENT: ANALYTICS\nREASONING: The user is requesting a forward-looking prediction or trend analysis.\nTHOUGHTS: Routing to Analytics Agent to trigger Scikit-Learn predictions."
        elif any(k in query_lower for k in ["whatsapp", "alert", "message", "sms", "notify", "எச்சரிக்கை", "செய்தி"]):
            routing_response = "AGENT: COMMUNICATION\nREASONING: The user wants to write a notification template or trigger a message alert.\nTHOUGHTS: Routing to Communication Agent for messaging draft."
        elif any(k in query_lower for k in ["supplier", "vendor", "procure", "purchase order", "po", "reorder", "கொள்முதல்", "விற்பனையாளர்"]):
            routing_response = "AGENT: SUPPLIER\nREASONING: The user is asking about supplier catalog, procurement, or reorder parameters.\nTHOUGHTS: Routing to Supplier Agent for procurement calculations."
        elif any(k in query_lower for k in ["customer", "loyalty", "repeat", "buyer", "clv", "aov", "segment", "spending", "visit", "bought together", "வாடிக்கையாளர்", "அடிக்கடி"]):
            routing_response = "AGENT: CUSTOMER\nREASONING: The query deals with customer retention, spending, segments, or profiles.\nTHOUGHTS: Routing to Customer Insights Agent."
        elif any(k in query_lower for k in ["strategy", "growth", "branding", "marketing", "promotion", "competitor", "attract", "retain", "sales improvement", "வளர்ச்சி", "விளம்பரம்"]):
            routing_response = "AGENT: STRATEGY\nREASONING: The query asks for business growth strategies or promotional suggestions.\nTHOUGHTS: Routing to Business Strategy Agent."
        else:
            routing_response = "AGENT: GENERAL\nREASONING: General customer assistance query.\nTHOUGHTS: General coordinator fallback."
  
    # Parse routing response
    agent = "GENERAL"
    reasoning = "General business query."
    thoughts = "Handling query via master coordinator."
    
    for line in routing_response.split('\n'):
        if line.startswith("AGENT:"):
            agent = line.replace("AGENT:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
        elif line.startswith("THOUGHTS:"):
            thoughts = line.replace("THOUGHTS:", "").strip()
  
    # 2. Run the selected agent with dynamically appended DB context
    db_context = ""
    if agent == "FINANCE":
        db_context = get_finance_db_context(workspace_dir)
        try:
            from forecasting.customer_analytics import get_customer_insights
            insights = get_customer_insights(workspace_dir)
            if not insights.get("is_empty", True):
                metrics = insights["metrics"]
                db_context += (
                    f"\nCustomer Context for Finance:\n"
                    f"- Total Customers: {metrics['total_customers']}\n"
                    f"- Repeat Buyer Rate: {metrics['repeat_customer_rate']}%\n"
                    f"- Average Order Value (AOV): Rs. {metrics['average_order_value']:.2f}\n"
                )
        except Exception:
            pass
        combined_context = f"{db_context}\n\n---\n\n{context}"
        response = run_finance_agent(query, combined_context, provider=provider)
    elif agent == "INVENTORY":
        db_context = get_inventory_db_context(workspace_dir)
        combined_context = f"{db_context}\n\n---\n\n{context}"
        response = run_inventory_agent(query, combined_context, provider=provider)
    elif agent == "ANALYTICS":
        db_context = get_analytics_db_context(query, workspace_dir)
        try:
            from forecasting.customer_analytics import get_customer_insights
            insights = get_customer_insights(workspace_dir)
            if not insights.get("is_empty", True):
                metrics = insights["metrics"]
                db_context += (
                    f"\nCustomer Retention & Growth Context:\n"
                    f"- Active Customers Count: {metrics['active_customers']}\n"
                    f"- Inactive Customers Count: {metrics['inactive_customers']}\n"
                    f"- Customer Lifetime Value (CLV): Rs. {metrics['customer_lifetime_value']:.2f}\n"
                    f"- Repeat Buyer Rate: {metrics['repeat_customer_rate']}%\n"
                )
        except Exception:
            pass
        combined_context = f"{db_context}\n\n---\n\n{context}"
        response = run_analytics_agent(query, combined_context, provider=provider)
    elif agent == "COMMUNICATION":
        db_context = f"{get_inventory_db_context(workspace_dir)}\n\n{get_finance_db_context(workspace_dir)}"
        try:
            from forecasting.customer_analytics import get_customer_insights
            insights = get_customer_insights(workspace_dir)
            if not insights.get("is_empty", True):
                db_context += f"\n\nCustomer Database Recommendations context:\n"
                for r in insights["recommendations"][:3]:
                    db_context += f"- Customer: {r['customer_name']} | Type: {r['type']} | Offer: {r['discount_offer']} | Detail: {r['message']}\n"
        except Exception:
            pass
        combined_context = f"{db_context}\n\n---\n\n{context}"
        response = run_communication_agent(query, combined_context, provider=provider)
    elif agent == "SUPPLIER":
        db_context = f"{get_supplier_db_context(workspace_dir)}\n\n{get_inventory_db_context(workspace_dir)}"
        combined_context = f"{db_context}\n\n---\n\n{context}"
        response = run_supplier_agent(query, combined_context, provider=provider)
    elif agent == "CUSTOMER":
        db_context = get_customer_db_context(workspace_dir)
        combined_context = f"{db_context}\n\n---\n\n{context}"
        response = run_customer_agent(query, combined_context, provider=provider)
    elif agent == "STRATEGY":
        # Load business profile
        profile = {}
        if workspace_dir:
            profile_path = os.path.join(workspace_dir, "business_profile.json")
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, "r", encoding="utf-8") as f:
                        profile = json.load(f)
                except Exception:
                    pass
            
            if not profile:
                from database import db as _db
                onb_doc = _db.onboarding.find_one({"workspace_id": os.path.basename(workspace_dir)}, {"_id": 0})
                if onb_doc:
                    profile = {
                        "businessName": onb_doc.get("businessName", "Small Business"),
                        "businessType": onb_doc.get("businessCategory", "Small Business"),
                        "businessSector": onb_doc.get("businessCategory", "Retail"),
                        "enterpriseCategory": "Micro",
                        "state": "Local",
                        "district": onb_doc.get("businessLocation", "Region"),
                        "annualTurnover": "Undisclosed",
                        "employeeCount": "N/A"
                    }
        
        if not profile:
            profile = {
                "businessName": "Small Business",
                "businessType": "Retail Shop",
                "businessSector": "Retail",
                "enterpriseCategory": "Micro",
                "state": "Local",
                "district": "Region",
                "annualTurnover": "Undisclosed",
                "employeeCount": "N/A"
            }
            
        goals = "attract"
        query_lower = query.lower()
        if "sales" in query_lower or "revenue" in query_lower or "price" in query_lower:
            goals = "sales"
        elif "retention" in query_lower or "loyalty" in query_lower or "repeat" in query_lower:
            goals = "retention"
        elif "brand" in query_lower or "identity" in query_lower:
            goals = "branding"
        elif "marketing" in query_lower or "social" in query_lower or "advertise" in query_lower:
            goals = "marketing"
            
        strategy_inputs = {
            "businessType": profile.get("businessType", "Small Business"),
            "targetAudience": "General Public",
            "goals": goals,
            "competitors": ""
        }
        
        lang = "english"
        if any(w in query_lower for w in ["தமிழ்", "tamil"]):
            lang = "tamil"
        elif any(w in query_lower for w in ["हिंदी", "hindi"]):
            lang = "hindi"
            
        response = generate_strategy(profile, strategy_inputs, language=lang)
    else:
        # General/Coordinator direct response - include basic general DB states
        db_context = f"{get_inventory_db_context(workspace_dir)}\n\n{get_finance_db_context(workspace_dir)}\n\n{get_supplier_db_context(workspace_dir)}\n\n{get_customer_db_context(workspace_dir)}"
        combined_context = f"{db_context}\n\n---\n\n{context}"
        general_prompt = (
            "You are AegisAI, an autonomous business copilot. Answer this general query "
            "directly using the provided active database summary and document context. "
            "Respond warmly to greetings. Format stock details or lists using markdown tables with exact database item names/suppliers."
        )
        response = call_llm(general_prompt, f"Context:\n{combined_context}\n\nQuery: {query}", provider=provider)
        
    return {
        "response": response,
        "agent": agent,
        "reasoning": reasoning,
        "thoughts": thoughts
    }
