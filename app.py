import os
import time
import datetime
import requests
import json
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# ==========================================
# 1. CONFIGURATION & KEYS
# ==========================================
TEST_MODE = False  # Set to True to run dummy news tests; set to False for live Finnhub polling

TEST_FILE_PATH = "./test_cases.txt"

def load_test_cases() -> list:
    if not os.path.exists(TEST_FILE_PATH):
        # Auto-create file if missing
        default_tests = [
            "Apple supplier Foxconn suffers factory disruption, delaying holiday iPhone shipments.",
            "Nvidia beats Q3 earnings expectations by 30% on skyrocketing AI data center demand."
        ]
        with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(default_tests))
        return default_tests

    with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
        # Load non-empty lines and ignore comment lines starting with '#'
        headlines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    return headlines

FINBERT_LOCAL_PATH = os.path.abspath("./finbert_financial_sentiment")

# Check if local model folder exists before attempting to load
if not os.path.exists(FINBERT_LOCAL_PATH):
    raise FileNotFoundError(
        f"Could not find model directory at: {FINBERT_LOCAL_PATH}\n"
        "Please make sure you unzipped your Colab model into a folder named 'finbert_model' "
        "inside your project directory."
    )


FINNHUB_API_KEY = "dan3sg1r01qn0fq9ktvgdan3sg1r01qn0fq9ku00"
WEBHOOK_URL = "https://discord.com/api/webhooks/1550778825821126711/veHHevu2vbrRS7PHtReC7SSWCHQCfO_vHKzKMghls5qreWNsVn-oEPUBYBrr3fY3GiwR"

ALERT_THRESHOLD = 0.80  # FinBERT confidence threshold (80%)
LABELS = ["Negative", "Neutral", "Positive"]
processed_news_ids = set()

# Detect Hardware Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running pipeline on device: {DEVICE.upper()}")

# ==========================================
# 2. STAGE 1: LOAD FINBERT MODEL
# ==========================================
print("\n[Stage 1] Loading local FinBERT model...")
finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_LOCAL_PATH)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_LOCAL_PATH).to(DEVICE)
finbert_model.eval()

def analyze_sentiment_finbert(headline: str) -> dict:
    inputs = finbert_tokenizer(
        headline, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    scores = {LABELS[i]: round(probs[i], 4) for i in range(len(LABELS))}
    top_label = max(scores, key=scores.get)
    return {"sentiment": top_label, "confidence": scores[top_label]}

# ==========================================
# 3. STAGE 2: LOAD LIGHTWEIGHT QWEN CAUSAL REASONER
# ==========================================
# 0.5B model is fast (~1GB download) and accurate for JSON output
REASONER_MODEL_NAME = os.path.abspath("./qwen_model")

reasoner_tokenizer = AutoTokenizer.from_pretrained(REASONER_MODEL_NAME, local_files_only=True)
reasoner_model = AutoModelForCausalLM.from_pretrained(REASONER_MODEL_NAME, local_files_only=True)

# Load model (uses float16 on GPU, float32 on CPU)
if DEVICE == "cuda":
    reasoner_model = AutoModelForCausalLM.from_pretrained(
        REASONER_MODEL_NAME,
        dtype=torch.float16,    # Fixed deprecation warning (replaced torch_dtype)
        device_map="auto"
    )
else:
    reasoner_model = AutoModelForCausalLM.from_pretrained(
        REASONER_MODEL_NAME,
        dtype=torch.float32,    # Fixed deprecation warning
        device_map="cpu"
    )

def analyze_stock_impact(headline: str) -> list:
    prompt = f"""Analyze this news headline and identify publicly traded companies/tickers affected.
Headline: "{headline}"

Respond ONLY with a valid JSON array of objects following this schema:
[
  {{
    "ticker": "TICKER",
    "company": "Company Name",
    "impact": "Bullish" or "Bearish" or "Neutral",
    "reasoning": "Short 1-sentence explanation of why/how this stock is affected."
  }}
]
JSON Output:"""

    messages = [
        {"role": "system", "content": "You output strictly valid JSON without markdown wrapping."},
        {"role": "user", "content": prompt}
    ]

    text_input = reasoner_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = reasoner_tokenizer([text_input], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = reasoner_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    response_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    raw_output = reasoner_tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()

    # Clean potential markdown formatting
    if "```" in raw_output:
        raw_output = raw_output.split("```")[1].replace("json", "").strip()

    try:
        return json.loads(raw_output)
    except Exception:
        return [{"ticker": "MARKET", "company": "General Market", "impact": "Neutral", "reasoning": raw_output}]


def send_discord_alert(headline: str, sentiment_res: dict, impact_res: list, news_release_time: str = None):
    """
    Sends structured Discord alerts containing stock metrics, release timestamps,
    and alert dispatch timestamps.
    """

    # 1. Determine Embed Color based on Sentiment
    sentiment = sentiment_res["sentiment"]
    if sentiment == "Positive":
        color = 0x2ECC71  # Vibrant Green
    elif sentiment == "Negative":
        color = 0xE74C3C  # Vibrant Red
    else:
        color = 0x95A5A6  # Neutral Gray

    # 2. Format Ticker & Causal Impact Block
    impact_fields = []
    for item in impact_res:
        ticker = item.get("ticker", "N/A")
        company = item.get("company", "N/A")
        direction = item.get("impact", "Neutral")
        reason = item.get("reasoning", "No detailed reasoning provided.")
        
        # Format Direction Symbol
        icon = "🟢" if direction == "Bullish" else "🔴" if direction == "Bearish" else "⚪"
        
        impact_fields.append(
            f"{icon} **${ticker}** ({company})\n"
            f"**Impact:** `{direction}`\n"
            f"**Reason:** {reason}"
        )
    
    impact_text = "\n\n".join(impact_fields) if impact_fields else "No specific tickers identified."

    # 3. Format Timestamps
    dispatched_time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 4. Construct Rich Discord Embed
    embed = {
        "title": "⚡ High-Impact Market Alert",
        "description": f"**Headline:** {headline}",
        "color": color,
        "fields": [
            {
                "name": "📈 Ticker & Causal Impact",
                "value": impact_text,
                "inline": False
            },
            {
                "name": "🎯 FinBERT Confidence",
                "value": f"`{sentiment_res['confidence'] * 100:.1f}% ({sentiment})`",
                "inline": False
            },
            {
                "name": "🚀 Dispatched to Channel",
                "value": f"`{dispatched_time_str}`",
                "inline": True
            }
        ],
        "footer": {
            "text": "Market Sentinel Agent | Rapid Real-Time Pipeline"
        }
    }

    payload = {"embeds": [embed]}

    try:
        res = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        if res.status_code in [200, 204]:
            print("  ✅ Discord Webhook alert dispatched successfully.")
        else:
            print(f"  ❌ Discord Webhook failed: HTTP {res.status_code}")
    except Exception as e:
        print(f"  ❌ Failed to send Discord alert: {e}")

# ==========================================
# 5. LIVE POLLING LOOP
# ==========================================
poll_counter = 0

def fetch_and_process_live_news():
    global poll_counter
    poll_counter += 1
    current_time = time.strftime("%H:%M:%S")
    
    # Visual Heartbeat Header
    print(f"\n[Pulse #{poll_counter:04d} | {current_time}] 💓 Connection Active — Querying Finnhub API...")
    
    url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
    
    try:
        start_request_time = time.time()
        response = requests.get(url, timeout=10)
        latency = round((time.time() - start_request_time) * 1000, 2)
        
        if response.status_code != 200:
            print(f"  ❌ API Response Warning: HTTP {response.status_code}")
            return

        news_items = response.json()
        print(f"  📡 API Stream Online | Received {len(news_items)} headlines ({latency}ms latency)")
        
        new_headlines_found = 0
        
        for item in reversed(news_items[:5]):
            news_id = item.get("id")
            headline = item.get("headline", "")
            
            if news_id in processed_news_ids or not headline:
                continue
                
            processed_news_ids.add(news_id)
            new_headlines_found += 1
            
            print(f"\n  --------------------------------------------------")
            print(f"  ⚡ NEW HEADLINE INGESTED: {headline}")
            
            # Stage 1: Fast FinBERT Filter
            sentiment_res = analyze_sentiment_finbert(headline)
            print(f"  -> Stage 1 FinBERT: {sentiment_res['sentiment']} ({sentiment_res['confidence'] * 100:.1f}%)")
            
            # Stage 2: Causal Reasoner (If high confidence signal)
            if sentiment_res["sentiment"] in ["Positive", "Negative"] and sentiment_res["confidence"] >= ALERT_THRESHOLD:
                print("  🚨 High Impact Signal! Invoking Stage 2 Causal Reasoner...")
                impact_res = analyze_stock_impact(headline)
                
                # Stage 3: Webhook Alert
                send_discord_alert(headline, sentiment_res, impact_res)
            else:
                print("  ℹ️ Filtered out (Neutral or below threshold).")
            print(f"  --------------------------------------------------")
            
        if new_headlines_found == 0:
            print("  💤 Stream Idle: No new unique headlines since last pulse.")

    except requests.exceptions.Timeout:
        print("  ⚠️ API Request Timed Out (Retrying next cycle...)")
    except Exception as e:
        print(f"  ❌ Stream Execution Error: {e}")

# ==========================================
# 6. DYNAMIC TEST SUITE RUNNER
# ==========================================
def run_test_suite():
    headlines = load_test_cases()

    
    print("\n" + "="*60)
    print(f"🧪 TEST MODE: Executing {len(headlines)} headlines from {TEST_FILE_PATH}")
    print("="*60)
    
    for idx, headline in enumerate(headlines, start=1):
        print(f"\n[Test Case {idx}/{len(headlines)}]: {headline}")
        print("-" * 50)
        
        # Stage 1: FinBERT Sentiment
        sentiment_res = analyze_sentiment_finbert(headline)
        print(f"  -> Stage 1 FinBERT: {sentiment_res['sentiment']} ({sentiment_res['confidence'] * 100:.1f}%)")
        
        # Stage 2: Causal Reasoner
        if sentiment_res["sentiment"] in ["Positive", "Negative"] and sentiment_res["confidence"] >= ALERT_THRESHOLD:
            print("  🚨 Signal Passed Threshold! Running Stage 2 Causal Reasoner...")
            impact_res = analyze_stock_impact(headline)
            
            print("  📊 Impact Analysis:")
            for item in impact_res:
                print(f"     • ${item.get('ticker', 'N/A')} ({item.get('company', 'N/A')}): {item.get('impact', 'N/A')}")
                print(f"       Reason: {item.get('reasoning', 'N/A')}")
            
            # Stage 3: Webhook Alert
                test_release_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                send_discord_alert(headline, sentiment_res, impact_res, news_release_time=test_release_time)
        else:
            print("  ℹ️ Filtered out (Neutral or below threshold).")
            
        print("-" * 50)
        time.sleep(1)

    print(f"\n✅ Finished processing all {len(headlines)} test cases!")


if __name__ == "__main__":
    print("\n🚀 Market Sentinel Agent Initialized")
    print(f"Mode             : {'🧪 TEST MODE' if TEST_MODE else '🔴 LIVE POLLING'}")
    print(f"FinBERT Device   : {DEVICE.upper()}")
    print("--------------------------------------------------")
    
    if TEST_MODE:
        run_test_suite()
    else:
        print("Starting 60-second live news polling cycle...\n")
        try:
            while True:
                fetch_and_process_live_news()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n🛑 Market Sentinel Agent stopped by user.")