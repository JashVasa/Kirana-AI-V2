"""Order Extraction, Catalog Matching (RAG), and Security Guardrail Engine.
Processes messy, informal Hinglish WhatsApp messages from Kirana store owners.
Features:
- Dual-mode extraction: Google Gemini Live API with automatic text model discovery,
  and a built-in deterministic fuzzy/token parser for 100% reliable offline testing.
- Ambiguity Resolver (e.g. 'cola' matching Thums Up vs Coca-Cola).
- Out-of-catalog detection (e.g. 'Doritos').
- Strict Security Guardrails: Prompt injection detection, price tampering prevention,
  and zero-hallucination catalog grounding.
"""
import re
import json
import os
import difflib
import requests

BASE_DIR = os.path.dirname(__file__)
CATALOG_PATH = os.path.join(BASE_DIR, "catalog.json")

def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

CATALOG = load_catalog()

# Number words mapping for Hinglish & English
NUMBER_WORDS = {
    "ek": 1, "aik": 1, "one": 1, "1": 1,
    "do": 2, "doh": 2, "two": 2, "2": 2,
    "teen": 3, "tin": 3, "three": 3, "3": 3,
    "char": 4, "chaar": 4, "four": 4, "4": 4,
    "paanch": 5, "panch": 5, "five": 5, "5": 5,
    "chhe": 6, "che": 6, "six": 6, "6": 6,
    "saat": 7, "sat": 7, "seven": 7, "7": 7,
    "aath": 8, "ath": 8, "eight": 8, "8": 8,
    "nau": 9, "no": 9, "nine": 9, "9": 9,
    "das": 10, "dass": 10, "ten": 10, "10": 10,
    "gyarah": 11, "eleven": 11, "11": 11,
    "barah": 12, "twelve": 12, "12": 12,
    "pandrah": 15, "fifteen": 15, "15": 15,
    "bees": 20, "twenty": 20, "20": 20,
    "pachees": 25, "twenty five": 25, "25": 25,
    "pachas": 50, "fifty": 50, "50": 50,
    "sau": 100, "hundred": 100, "100": 100
}

KNOWN_UNITS = [
    "peti", "petee", "crate", "box", "carton", "dabba", "dabbe",
    "bori", "bory", "bag", "sack", "tin", "packet", "pkt", "pkts",
    "bottle", "bottles", "piece", "pcs", "kilo", "kg", "litre", "ltr"
]

AMBIGUOUS_KEYWORDS = {
    "cola": {
        "candidate_skus": ["SKU001", "SKU009"],
        "clarification_prompt": "Bhai, 'cola' me kaunsa brand chahiye: Thums Up 500ml ya Coca-Cola 500ml?"
    },
    "cold drink": {
        "candidate_skus": ["SKU001", "SKU009"],
        "clarification_prompt": "Bhai, cold drink me Thums Up 500ml chahiye ya Coca-Cola 500ml?"
    },
    "namak": {
        "candidate_skus": ["SKU002"],
        "clarification_prompt": "Bhai, Tata Salt 1kg bhej dein ya koi aur brand?"
    }
}

KNOWN_OUT_OF_CATALOG = [
    "doritos", "lays", "kurkure", "bingo", "dairy milk", "kitkat", "5 star",
    "red bull", "sting", "frooti", "real juice", "bournvita", "horlicks",
    "pepsodent", "colgate", "rin", "wheel", "tide", "harpic", "good knight",
    "all out", "vicks", "crocin", "cigarettes", "gold flake", "classic"
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) (instructions|rules)",
    r"system (override|prompt|admin)",
    r"dispatch .* (price 0|for free|zero rupees|free of cost)",
    r"bypass (validation|authorization|rules)",
    r"you are now (an admin|in debug mode|a root user)",
    r"developer mode",
    r"authorized by (distributor|owner|director) without payment"
]

def check_prompt_injection(text: str) -> tuple[bool, str]:
    """Scans untrusted user order text for prompt injection / jailbreak attempts."""
    clean_text = text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, clean_text):
            return True, f"Security Alert: Malicious prompt injection pattern detected ('{pattern}'). Order halted."
    return False, ""

def match_catalog_item(raw_mention: str) -> tuple[dict | None, float, list]:
    """Matches a free-text item mention to catalog SKUs using alias matching and string similarity."""
    raw_lower = raw_mention.lower().strip()
    
    # 1. Exact alias match
    for prod in CATALOG:
        for alias in prod["aliases"]:
            if alias in raw_lower:
                return prod, 1.0, []
                
    # 2. Fuzzy match against all aliases and product names
    best_prod = None
    best_score = 0.0
    
    for prod in CATALOG:
        candidates = [prod["name"].lower()] + [a.lower() for a in prod["aliases"]]
        for cand in candidates:
            score = difflib.SequenceMatcher(None, raw_lower, cand).ratio()
            # Partial substring check
            if cand in raw_lower or raw_lower in cand:
                score = max(score, 0.75)
            if score > best_score:
                best_score = score
                best_prod = prod
                
    if best_score >= 0.65:
        return best_prod, best_score, []
        
    return None, best_score, []

def deterministic_parse(message: str) -> dict:
    """Robust local parser for kirana WhatsApp messages.
    Handles multi-item orders, units, quantities, ambiguities, and out-of-catalog items.
    """
    is_injected, injection_reason = check_prompt_injection(message)
    if is_injected:
        return {
            "status": "BLOCKED_BY_GUARDRAIL",
            "message": message,
            "error": injection_reason,
            "items": [],
            "clarifications_needed": [injection_reason],
            "out_of_catalog_items": [],
            "total_amount": 0.0
        }
        
    clean_msg = message.lower()
    
    # Check for ambiguous generic keywords first
    ambiguities = []
    for ambig_key, ambig_data in AMBIGUOUS_KEYWORDS.items():
        if re.search(r"\b" + re.escape(ambig_key) + r"\b", clean_msg):
            # Check if an exact specific brand was already named
            has_specific = ("thums up" in clean_msg or "coca cola" in clean_msg or "coke" in clean_msg)
            if not has_specific and ambig_key in ["cola", "cold drink"]:
                ambiguities.append({
                    "keyword": ambig_key,
                    "candidates": ambig_data["candidate_skus"],
                    "question": ambig_data["clarification_prompt"]
                })
                
    # Check for known out-of-catalog items
    out_of_catalog = []
    for ooc in KNOWN_OUT_OF_CATALOG:
        if re.search(r"\b" + re.escape(ooc) + r"\b", clean_msg):
            out_of_catalog.append({
                "raw_mention": ooc.title(),
                "reason": f"'{ooc.title()}' is not stocked in distributor FMCG catalog."
            })
            
    # Split text into item segments by conjunctions (aur, and, comma, plus, newline)
    raw_segments = re.split(r",|aur|\band\b|\+|\n|bhej\s+do|bhej\s+dena", clean_msg)
    
    parsed_items = []
    clarifications = [a["question"] for a in ambiguities]
    
    for seg in raw_segments:
        seg = seg.strip()
        if not seg or len(seg) < 3:
            continue
            
        # Check if segment matches an out-of-catalog item
        is_ooc = any(ooc["raw_mention"].lower() in seg for ooc in out_of_catalog)
        if is_ooc:
            continue
            
        # Extract quantity
        qty = None
        # Try numeric digits first
        num_match = re.search(r"\b(\d+)\b", seg)
        if num_match:
            qty = int(num_match.group(1))
        else:
            # Try Hinglish number words
            for word, val in NUMBER_WORDS.items():
                if re.search(r"\b" + re.escape(word) + r"\b", seg):
                    qty = val
                    break
                    
        # Extract unit
        matched_unit = None
        for u in KNOWN_UNITS:
            if re.search(r"\b" + re.escape(u) + r"\b", seg):
                matched_unit = u
                break
                
        # Match product
        prod, score, _ = match_catalog_item(seg)
        
        if prod:
            # If quantity is missing or vague (e.g. "thoda tel")
            if qty is None:
                clarifications.append(f"Bhai, '{prod['name']}' kitna {prod['primary_unit']} chahiye? Quantity batayein.")
                continue
                
            unit_to_use = matched_unit if matched_unit else prod["primary_unit"]
            line_total = qty * prod["unit_price"]
            
            parsed_items.append({
                "sku": prod["sku"],
                "product_name": prod["name"],
                "category": prod["category"],
                "pack_size": prod["pack_size"],
                "quantity": qty,
                "unit": unit_to_use,
                "unit_price": prod["unit_price"],
                "line_total": line_total,
                "confidence": round(score, 2)
            })
            
    total_amount = sum(item["line_total"] for item in parsed_items)
    
    if clarifications:
        status = "NEEDS_CLARIFICATION"
    elif out_of_catalog and not parsed_items:
        status = "OUT_OF_CATALOG_ONLY"
    elif out_of_catalog and parsed_items:
        status = "PARTIALLY_VALIDATED"
    elif parsed_items:
        status = "VALIDATED"
    else:
        status = "UNKNOWN_OR_EMPTY"
        clarifications.append("Order samajh nahi aaya. Kripya product ka naam aur peti/dabba quantity likhein.")
        
    return {
        "status": status,
        "message": message,
        "items": parsed_items,
        "clarifications_needed": clarifications,
        "out_of_catalog_items": out_of_catalog,
        "total_amount": total_amount
    }

def get_supported_gemini_models(api_key: str) -> list[tuple[str, str]]:
    """Dynamically query Google's ModelService to find TEXT models available for this specific API key."""
    for ver in ["v1beta", "v1"]:
        try:
            url = f"https://generativelanguage.googleapis.com/{ver}/models?key={api_key}"
            r = requests.get(url, timeout=7)
            if r.status_code == 200:
                data = r.json()
                valid = []
                for m in data.get("models", []):
                    methods = m.get("supportedGenerationMethods", [])
                    name = m.get("name", "").replace("models/", "")
                    if "generateContent" in methods and not any(bad in name.lower() for bad in ["tts", "audio", "embed", "imagen"]):
                        valid.append((ver, name))
                if valid:
                    return valid
        except Exception:
            continue
    return [("v1beta", "gemini-2.0-flash"), ("v1beta", "gemini-1.5-flash")]

def parse_with_gemini(message: str, api_key: str) -> dict:
    """Uses Gemini to parse informal Kirana text into structured JSON, falling back to deterministic parser."""
    is_injected, injection_reason = check_prompt_injection(message)
    if is_injected:
        return deterministic_parse(message)
        
    catalog_summary = "\n".join([f"- {p['sku']}: {p['name']} ({', '.join(p['aliases'][:4])}) | Unit: {p['primary_unit']} | Price: Rs {p['unit_price']}" for p in CATALOG])
    
    system_prompt = (
        "You are an expert FMCG Kirana order extraction assistant for a distributor in Mumbai/Kalyan.\n"
        "Your task: Convert messy, informal Hinglish WhatsApp messages into structured JSON.\n\n"
        "DISTRIBUTOR CATALOG (STRICT SOURCE OF TRUTH):\n"
        f"{catalog_summary}\n\n"
        "RULES:\n"
        "1. Never invent SKUs, prices or products. Prices must come strictly from the catalog.\n"
        "2. If an item is ambiguous (e.g. 'cola' could be Thums Up or Coca-Cola), do NOT guess. Add a clarification question.\n"
        "3. If an item is not in catalog (e.g. Doritos), put it in 'out_of_catalog'.\n"
        "4. If quantity or unit is missing, request clarification.\n"
        "5. Output valid JSON ONLY matching this schema:\n"
        "{\n"
        "  \"items\": [{\"sku\": \"...\", \"product_name\": \"...\", \"quantity\": 1, \"unit\": \"peti\", \"unit_price\": 720.0, \"line_total\": 720.0}],\n"
        "  \"clarifications\": [\"Bhai, cola me Thums Up chahiye ya Coca-Cola?\"],\n"
        "  \"out_of_catalog\": [\"Doritos\"]\n"
        "}"
    )
    
    models = get_supported_gemini_models(api_key)
    for ver, model_name in models:
        url = f"https://generativelanguage.googleapis.com/{ver}/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": system_prompt + f"\n\nORDER MESSAGE: \"{message}\"\n\nJSON Output:"}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 400}
        }
        try:
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                raw_out = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean markdown backticks if returned
                if "```json" in raw_out:
                    raw_out = raw_out.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_out:
                    raw_out = raw_out.split("```")[1].split("```")[0].strip()
                parsed_json = json.loads(raw_out)
                
                # Ground items against catalog prices
                validated_items = []
                for it in parsed_json.get("items", []):
                    sku = it.get("sku")
                    cat_prod = next((p for p in CATALOG if p["sku"] == sku), None)
                    if cat_prod:
                        qty = int(it.get("quantity", 1))
                        validated_items.append({
                            "sku": cat_prod["sku"],
                            "product_name": cat_prod["name"],
                            "category": cat_prod["category"],
                            "pack_size": cat_prod["pack_size"],
                            "quantity": qty,
                            "unit": it.get("unit", cat_prod["primary_unit"]),
                            "unit_price": cat_prod["unit_price"],
                            "line_total": qty * cat_prod["unit_price"],
                            "confidence": 0.95
                        })
                        
                clarifications = parsed_json.get("clarifications", [])
                out_of_cat = [{"raw_mention": x, "reason": f"'{x}' is not stocked in distributor catalog."} for x in parsed_json.get("out_of_catalog", [])]
                
                status = "VALIDATED"
                if clarifications:
                    status = "NEEDS_CLARIFICATION"
                elif out_of_cat and not validated_items:
                    status = "OUT_OF_CATALOG_ONLY"
                elif out_of_cat:
                    status = "PARTIALLY_VALIDATED"
                    
                total_amt = sum(x["line_total"] for x in validated_items)
                return {
                    "status": status,
                    "message": message,
                    "items": validated_items,
                    "clarifications_needed": clarifications,
                    "out_of_catalog_items": out_of_cat,
                    "total_amount": total_amt,
                    "engine_used": f"Gemini ({model_name})"
                }
        except Exception:
            continue
            
    # Fallback to deterministic parser
    fallback_res = deterministic_parse(message)
    fallback_res["engine_used"] = "Deterministic Hybrid Engine (Fallback)"
    return fallback_res
