import os
import re
import requests
import pandas as pd
import config


def _call_groq_with_key(api_key: str, system_instruction: str, prompt: str, temperature: float) -> str:
    """Raw Groq API call using candidate models."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # Candidate models supported by Groq (env override first)
    candidate_models = [
        os.getenv("GROQ_MODEL", "").strip(),
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "groq/compound-mini",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    candidate_models = [m for m in candidate_models if m]

    last_error = None
    for model_name in candidate_models:
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                response.raise_for_status()
            else:
                last_error = f"{response.status_code}: {response.text}"
                continue
        except requests.exceptions.HTTPError:
            raise
        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"Groq API call failed across candidate models. Last error: {last_error}")


def call_groq_llm(system_instruction: str, prompt: str, temperature: float = 0.2) -> str:
    """Calls Groq with primary key, falls back to backup key on rate limit (429)."""
    primary = config.get_groq_key()
    backup = config.get_groq_backup_key()

    if not primary and not backup:
        raise ValueError("No Groq API key configured. Set GROQ_API_KEY in .env")

    if primary:
        try:
            return _call_groq_with_key(primary, system_instruction, prompt, temperature)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print("[LLM] Primary Groq key rate limited (429). Switching to backup key.")
            else:
                raise

    if backup:
        print("[LLM] Using backup Groq key.")
        return _call_groq_with_key(backup, system_instruction, prompt, temperature)

    raise ValueError("Both Groq keys exhausted or unavailable.")


def call_llm(system_instruction: str, prompt: str, temperature: float = 0.2, provider: str = "groq") -> str:
    """Unified LLM call — uses Groq with automatic fallback to local rule-based engine on failure."""
    try:
        return call_groq_llm(system_instruction, prompt, temperature)
    except Exception as e:
        print(f"[LLM] Live model call failed ({e}). Utilizing offline local engine fallback.")
        return fallback_local_agent(system_instruction, prompt)



def fallback_local_agent(system_instruction: str, prompt: str) -> str:
    """
    A data-driven offline agent fallback. It parses the system instruction to route
    to the correct virtual agent, and uses local data files to provide high-fidelity answers.
    """
    prompt_lower = prompt.lower()
    instruction_lower = system_instruction.lower()
    
    # Load files if they exist
    tx_df = pd.DataFrame()
    inv_df = pd.DataFrame()
    tx_path = os.path.join(config.DATA_DIR, 'transactions.csv')
    inv_path = os.path.join(config.DATA_DIR, 'inventory.csv')
    
    if os.path.exists(tx_path):
        tx_df = pd.read_csv(tx_path)
    if os.path.exists(inv_path):
        inv_df = pd.read_csv(inv_path)
        
    # Check language (English vs Tamil)
    is_tamil = any(word in prompt_lower or word in instruction_lower for word in ["வணக்கம்", "விற்பனை", "பொருள்", "இருப்பு", "கணிப்பு", "அடுத்த", "மாதம்"])
    
    # 1. Master Coordinator Routing Fallback
    if "coordinator" in instruction_lower:
        if any(k in prompt_lower for k in ["revenue", "sales", "finance", "expense", "profit", "earn", "விற்பனை", "வருவாய்", "இலாபம்"]):
            return "AGENT: FINANCE\nREASONING: The query deals with financial performance and sales amounts.\nTHOUGHTS: Routing to Finance Agent for ledger analysis."
        elif any(k in prompt_lower for k in ["stock", "inventory", "reorder", "restock", "low", "பொருள்", "இருப்பு", "தேவை"]):
            return "AGENT: INVENTORY\nREASONING: The query asks about stock status, quantities, or suppliers.\nTHOUGHTS: Routing to Inventory Agent to check warehouse logs."
        elif any(k in prompt_lower for k in ["predict", "forecast", "future", "next month", "கணிப்பு", "அடுத்த மாதம்"]):
            return "AGENT: ANALYTICS\nREASONING: The user is requesting a forward-looking prediction or trend analysis.\nTHOUGHTS: Routing to Analytics Agent to trigger Scikit-Learn predictions."
        elif any(k in prompt_lower for k in ["whatsapp", "alert", "message", "sms", "notify", "எச்சரிக்கை", "செய்தி"]):
            return "AGENT: COMMUNICATION\nREASONING: The user wants to write a notification template or trigger a message alert.\nTHOUGHTS: Routing to Communication Agent for messaging draft."
        elif any(k in prompt_lower for k in ["customer", "loyalty", "repeat", "buyer", "clv", "aov", "segment", "spending", "visit", "bought together", "வாடிக்கையாளர்", "அடிக்கடி"]):
            return "AGENT: CUSTOMER\nREASONING: The query deals with customer retention, spending, segments, or profiles.\nTHOUGHTS: Routing to Customer Insights Agent."
        else:
            return "AGENT: GENERAL\nREASONING: General customer assistance query.\nTHOUGHTS: General coordinator fallback."

    # 2. COMMUNICATION Agent - Draft WhatsApp alerts
    elif "communication agent" in instruction_lower:
        # Extract product details from the trigger prompt
        prod_name_match = re.search(r"product:\s*(.*?)\s*\(ID:", prompt, re.IGNORECASE)
        prod_id_match = re.search(r"ID:\s*([a-zA-Z0-9]+)\)", prompt, re.IGNORECASE)
        curr_stock_match = re.search(r"Current Stock:\s*(\d+)", prompt, re.IGNORECASE)
        reorder_limit_match = re.search(r"reorder limit:\s*(\d+)", prompt, re.IGNORECASE)
        velocity_match = re.search(r"selling\s*([\d\.]+)\s*units", prompt, re.IGNORECASE)
        days_remaining_match = re.search(r"run out in\s*([\d\.]+)\s*days", prompt, re.IGNORECASE)
        supplier_match = re.search(r"Supplier is\s*(.*?)\.\s*(?:Recommend|$)", prompt, re.IGNORECASE)
        recommend_match = re.search(r"Recommend restocking\s*(\d+)\s*units", prompt, re.IGNORECASE)

        if prod_name_match:
            prod_name = prod_name_match.group(1).strip()
            prod_id = prod_id_match.group(1).strip() if prod_id_match else "N/A"
            curr_stock = curr_stock_match.group(1).strip() if curr_stock_match else "0"
            reorder_limit = reorder_limit_match.group(1).strip() if reorder_limit_match else "0"
            velocity = velocity_match.group(1).strip() if velocity_match else "0.0"
            days_remaining = days_remaining_match.group(1).strip() if days_remaining_match else "0"
            supplier = supplier_match.group(1).strip() if supplier_match else "Supplier"
            recommend = recommend_match.group(1).strip() if recommend_match else "0"

            if is_tamil:
                return (
                    f"🚨 *குறைந்த இருப்பு எச்சரிக்கை* 🚨\n\n"
                    f"📦 *பொருள்:* {prod_name} (ID: {prod_id})\n"
                    f"⚠️ *தற்போதைய இருப்பு:* {curr_stock} அலகுகள் (வரம்பு: {reorder_limit})\n"
                    f"📉 *விற்பனை வேகம்:* ஒரு நாளைக்கு {velocity} அலகுகள் (இருப்பு {days_remaining} நாட்களில் தீர்ந்துவிடும்)\n"
                    f"🏭 *வழங்குநர்:* {supplier}\n\n"
                    f"💡 *பரிந்துரை:* இருப்புத் தட்டுப்பாட்டைத் தவிர்க்க உடனடியாக *{recommend} அலகுகள்* ஆர்டர் செய்யவும்."
                )
            else:
                return (
                    f"🚨 *LOW STOCK ALERT* 🚨\n\n"
                    f"📦 *Product:* {prod_name} (ID: {prod_id})\n"
                    f"⚠️ *Current Stock:* {curr_stock} units (Reorder Limit: {reorder_limit})\n"
                    f"📉 *Velocity:* {velocity} units/day (Exhaustion in {days_remaining} days)\n"
                    f"🏭 *Supplier:* {supplier}\n\n"
                    f"💡 *Recommendation:* Restock *{recommend} units* immediately to prevent stockout."
                )
        else:
            if is_tamil:
                return f"🔔 *செய்தி வரைவு:*\n\nவணக்கம், {prompt}"
            else:
                return f"🔔 *Draft Message:*\n\nDear Owner,\n\nHere is the information you requested regarding: {prompt}"

    # 3. ANALYTICS Agent - Dynamic Scikit-Learn sales forecast explanation
    elif "analytics agent" in instruction_lower:
        from forecasting.predictor import forecast_sales, predict_inventory_exhaustion
        # Extract product ID if mentioned in prompt
        prod_id = None
        for pid in ["P101", "P102", "P103", "P104", "P105", "P106", "P107", "P108"]:
            if pid.lower() in prompt_lower:
                prod_id = pid
                break
                
        forecast_res = forecast_sales(product_id=prod_id, days_ahead=30)
        depletion_data = predict_inventory_exhaustion()
        
        growth = forecast_res.get("growth_rate", 8.52)
        total_sales = forecast_res.get("total_forecasted_sales", 0.0)
        prod_label = forecast_res.get("product_name") or "Total Business"
        
        risks = [f"{item['ProductName']} ({item['DaysRemaining']} days left)" for item in depletion_data if item["Status"] in ["Low Stock", "Approaching Outage"]]
        risk_str = ", ".join(risks[:2]) if risks else "No critical stock exhaustion risks detected."
        
        if is_tamil:
            return (
                f"### 📈 விற்பனை கணிப்பு (ஆஃப்லைன் பகுப்பாய்வு முகவர்)\n"
                f"எங்கள் இயந்திர கற்றல் மாதிரி (Scikit-Learn) {prod_label} விற்பனையில் அடுத்த 30 நாட்களில் **{growth}%** வளர்ச்சியை கணிக்கிறது.\n"
                f"- **அடுத்த 30 நாட்களின் கணிக்கப்பட்ட மொத்த வருவாய்:** Rs. {total_sales:,.2f}\n"
                f"- **இருப்பு தீரும் அபாயங்கள்:** {risk_str}\n\n"
                f"விளக்கப்படங்களை விரிவாகக் காண **Predictive Analytics** தாவலுக்குச் செல்லவும்."
            )
        else:
            return (
                f"### 📈 Predictive Forecast (Offline Analytics Agent)\n"
                f"Using our Scikit-Learn linear regression model, we predict a **{growth}% trend** for {prod_label} sales over the next 30 days.\n"
                f"- **Forecasted Revenue (Next 30 Days):** Rs. {total_sales:,.2f}\n"
                f"- **Exhaustion Risks:** {risk_str}\n\n"
                f"Please head to the **Predictive Analytics** tab to view the visual trends."
            )

    # 4. FINANCE Agent - Sales & expense summary
    elif "finance agent" in instruction_lower:
        if tx_df.empty:
            return "No transaction history available." if not is_tamil else "விற்பனைத் தரவு எதுவும் இல்லை."
            
        sales = tx_df[tx_df['Type'] == 'Sale']['Amount'].sum()
        expenses = tx_df[tx_df['Type'] == 'Expense']['Amount'].sum()
        profit = sales - expenses
        
        cat_sales = tx_df[tx_df['Type'] == 'Sale'].groupby('Category')['Amount'].sum().to_dict()
        cat_str = "\n".join([f"- **{cat}**: Rs. {amt:,.2f}" for cat, amt in cat_sales.items()])
        
        if is_tamil:
            return (
                f"### 📊 நிதிப் பகுப்பாய்வு (ஆஃப்லைன் நிதி முகவர்)\n"
                f"கடந்த 6 மாதங்களின் நிதி நிலவரம்:\n"
                f"- **மொத்த விற்பனை (வருவாய்):** Rs. {sales:,.2f}\n"
                f"- **மொத்த செலவுகள்:** Rs. {expenses:,.2f}\n"
                f"- **நிகர இலாபம்:** Rs. {profit:,.2f}\n\n"
                f"**வகை வாரியான விற்பனை விவரம்:**\n{cat_str}\n\n"
                f"*குறிப்பு: இந்தத் தொகுப்பு ஆஃப்லைன் பரிவர்த்தனை தரவிலிருந்து கணக்கிடப்பட்டது.*"
            )
        else:
            return (
                f"### 📊 Financial Analysis (Offline Finance Agent)\n"
                f"Here is the summary of your financial data for the last 6 months:\n"
                f"- **Total Revenue (Sales):** Rs. {sales:,.2f}\n"
                f"- **Total Expenses:** Rs. {expenses:,.2f}\n"
                f"- **Net Profit:** Rs. {profit:,.2f}\n\n"
                f"**Category Breakdown:**\n{cat_str}\n\n"
                f"*Note: Compiled locally from transactions database.*"
            )

    # 5. INVENTORY Agent - Low stock reporting
    elif "inventory agent" in instruction_lower:
        if inv_df.empty:
            return "No inventory database found." if not is_tamil else "பொருட்களின் இருப்புத் தரவு எதுவும் இல்லை."
            
        from forecasting.predictor import predict_inventory_exhaustion
        depletion_data = predict_inventory_exhaustion()
        low_stock_str = ""
        
        low_stock = [item for item in depletion_data if item["Status"] == "Low Stock"]
        if low_stock:
            for item in low_stock:
                low_stock_str += f"- **{item['ProductName']}** (ID: {item['ProductID']}) | Current Stock: **{item['CurrentStock']} units** (Reorder limit: {item['ReorderLevel']}) | Supplier: {item['Supplier']}\n"
        else:
            low_stock_str = "All items have healthy stock levels!" if not is_tamil else "அனைத்து பொருட்களும் போதுமான இருப்புடன் உள்ளன!"
            
        if is_tamil:
            return (
                f"### 📦 இருப்பு பகுப்பாய்வு (ஆஃப்லைன் இருப்பு முகவர்)\n"
                f"**குறைந்த இருப்பு எச்சரிக்கை (உடனே ஆர்டர் செய்யவும்):**\n"
                f"{low_stock_str}\n"
                f"*குறிப்பு: ஆஃப்லைன் இருப்புப் பதிவுகளிலிருந்து நேரடியாகத் தொகுக்கப்பட்டது.*"
            )
        else:
            return (
                f"### 📦 Inventory Analysis (Offline Inventory Agent)\n"
                f"**Low Stock Alert (Immediate Reorder Required):**\n"
                f"{low_stock_str}\n"
                f"*Note: Compiled directly from active inventory records.*"
            )

    # 5b. CUSTOMER Agent
    elif "customer agent" in instruction_lower:
        from agents.customer_agent import fallback_customer_agent
        return fallback_customer_agent(prompt, None)

    # 6. Keyword and Language fallback
    if any(k in prompt_lower for k in ["revenue", "sales", "finance", "expense", "profit", "earn", "விற்பனை", "வருவாய்", "இலாபம்"]):
        # Finance keyword fallback
        if tx_df.empty: return "No transactions found."
        sales = tx_df[tx_df['Type'] == 'Sale']['Amount'].sum()
        expenses = tx_df[tx_df['Type'] == 'Expense']['Amount'].sum()
        profit = sales - expenses
        if is_tamil:
            return f"விற்பனை: Rs. {sales:,.2f}, செலவு: Rs. {expenses:,.2f}, இலாபம்: Rs. {profit:,.2f}."
        return f"Sales: Rs. {sales:,.2f}, Expenses: Rs. {expenses:,.2f}, Profit: Rs. {profit:,.2f}."
        
    elif any(k in prompt_lower for k in ["stock", "inventory", "reorder", "restock", "low", "பொருள்", "இருப்பு", "தேவை"]):
        # Inventory keyword fallback
        if inv_df.empty: return "No stock items."
        low_count = len(inv_df[inv_df['StockLevel'] <= inv_df['ReorderLevel']])
        if is_tamil:
            return f"குறைந்த இருப்புடன் {low_count} பொருட்கள் உள்ளன."
        return f"There are {low_count} low stock items in inventory."
        
    elif any(k in prompt_lower for k in ["predict", "forecast", "future", "next month", "கணிப்பு", "அடுத்த மாதம்"]):
        # Analytics keyword fallback
        from forecasting.predictor import forecast_sales
        f_res = forecast_sales(days_ahead=30)
        growth = f_res.get("growth_rate", 8.52)
        total_sales = f_res.get("total_forecasted_sales", 0.0)
        if is_tamil:
            return f"அடுத்த 30 நாட்களின் விற்பனை கணிப்பு: Rs. {total_sales:,.2f} ({growth}% வளர்ச்சி)."
        return f"30-day sales forecast: Rs. {total_sales:,.2f} (growth: {growth}%)."
        
    elif any(k in prompt_lower for k in ["customer", "loyalty", "repeat", "buyer", "clv", "aov", "segment", "spending", "visit", "bought together", "வாடிக்கையாளர்", "அடிக்கடி"]):
        from agents.customer_agent import fallback_customer_agent
        return fallback_customer_agent(prompt, None)

    # Standard Chat greetings and fallback
    if is_tamil:
        return (
            f"வணக்கம்! நான் ஏஜிஸ்ஏஐ (AegisAI) வணிக உதவியாளர்.\n\n"
            f"நீங்கள் என்னிடம் பின்வருவனவற்றைக் கேட்கலாம்:\n"
            f"1. **நிதி நிலைமை:** 'கடந்த மாத வருவாய் எவ்வளவு?'\n"
            f"2. **இருப்பு விவரம்:** 'எந்த பொருட்கள் குறைவாக உள்ளன?'\n"
            f"3. **விற்பனை கணிப்பு:** 'அடுத்த மாத விற்பனை எவ்வாறு இருக்கும்?'\n\n"
            f"நான் உங்களுக்கு உதவ தயாராக உள்ளேன்!"
        )
    else:
        return (
            f"Hello! I am AegisAI, your Multilingual Autonomous Business Copilot.\n\n"
            f"How can I help you manage your SME today? Here are some topics you can ask me:\n"
            f"- **Finance & Revenue**: 'What is my current profit margin?' or 'Summarize my weekly sales.'\n"
            f"- **Inventory & Supply**: 'Which products are low in stock?' or 'Who supplies Basmati Rice?'\n"
            f"- **Forecasting**: 'Will sales grow next month?' or 'Give me stock predictions.'\n"
            f"- **WhatsApp**: 'Send a low stock alert to my phone.'\n\n"
            f"*Currently running in local data-driven fallback mode. Add a Gemini API key in the sidebar for full conversational abilities.*"
        )
