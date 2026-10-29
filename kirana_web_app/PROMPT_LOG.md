# Kirana Order Assistant — Architecture, Guardrails & Edge Cases

## 1. Machine Learning: Logistic Regression vs. EWMA Heuristic
### Motivation: Why Replace Single EWMA with Logistic Regression?
While an Exponentially Weighted Moving Average (EWMA) tracks recent ordering recency, it collapses all store behavior into a single smoothed scalar. In real retail replenishment, reordering decisions are multi-dimensional:
- An item ordered 3 days ago might not be needed if its reorder interval is 14 days, but an item missing for 8 days on a 5-day cycle is urgently at risk of stockout.
- High-volume staple SKUs (e.g. Atta, Oil) have different replenishment dynamics than slow-moving items (e.g. Toilet Cleaner).

### Mathematical Formulation
The reorder probability for store $k$, SKU $i$ at cycle $t$ is modeled as:
$$P(\text{reorder} = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x})}}$$

Where feature vector $\mathbf{x} = [1, x_1, x_2, x_3, x_4, x_5]^T$:
1. **$x_1$ (Cycle Ratio):** Overdue factor calculated as $\text{cycle\_elapsed} / \overline{\text{order\_interval}}$. Ratios $> 1.0$ indicate the SKU is past its regular replenishment date.
2. **$x_2$ (In Last Order):** Binary indicator ($1$ if ordered at $t-1$, $0$ otherwise).
3. **$x_3$ (Short-Term Frequency):** Purchase rate over the previous 5 ordering cycles ($F_5 \in [0, 1]$).
4. **$x_4$ (Lifetime Frequency):** Historical purchase rate across all logged cycles ($F_{\text{all}} \in [0, 1]$).
5. **$x_5$ (EWMA Momentum):** Exponentially smoothed trend ($s_t = 0.3 y_t + 0.7 s_{t-1}$).
6. **$x_0$ (Intercept / Bias):** Baseline log-odds.

### Optimization & Results
- **Training:** Regularized Batch Gradient Descent on historical orders ($t = 0 \dots 25$) with L2 regularization penalty ($\lambda = 0.01$).
- **Test Performance ($t = 26 \dots 35$, 210 decisions):**
  - **Baseline (Last Order):** Recall 63.6%, F1 63.3%, Expected Risk Cost: ₹6,670
  - **EWMA (Threshold ≥ 0.55):** Recall 64.4%, F1 64.7%, Expected Risk Cost: ₹6,505
  - **Logistic Regression (ML):** **Recall 93.2%, F1 72.4%, Expected Risk Cost: ₹1,580** (over 75% reduction in stockout risk!)

---

## 2. RAG (Retrieval-Augmented Generation) for Message Parsing
### Motivation
Directly prompting an LLM with raw WhatsApp text or hardcoding strict regexes leads to two critical failure modes in Indian Kirana commerce:
1. **Hallucination:** Free-form generation invents non-existent pack sizes, unauthorized discounts, or out-of-stock brands.
2. **Dialect Misalignment:** Informal Hinglish nicknames (*"toofan"* for Thums Up, *"sarson tel"* for Dhara mustard oil, *"dabba"* for 5L tin) require grounded knowledge retrieval.

### RAG Pipeline Design
1. **Query Chunking & Intent Extraction:** Normalizes informal Hindi number words (`aadha`, `paanch`, `do`) and packaging units.
2. **Dense & Lexical Vector Retriever (The "R"):** Indexes 68 FMCG SKUs with brand aliases, category tokens, and pack signatures.
3. **Disambiguation & Guardrail Layer (The "AG"):**
   - Out-of-Catalog Detector: Blocks ungrounded items (e.g. *"nutella"*, *"dove soap"*).
   - Ambiguity Disambiguator: Emits interactive chips when candidate scores are tied.
   - Grounded Resolver: Maps verified SKUs directly to catalog pricing and case limits.

---

## 3. Edge Case Defenses & Business Guardrails

### Edge Case 1: 0.5 Pack / Fractional Pack Block
- **Business Rule:** In consumer retail distribution, packets/pouches (`pcs`, `packet`, `pkt`, `bag`) are indivisible units ($q \in \mathbb{Z}^+, q \ge 1$). A distributor cannot rip open a sealed consumer pack to deliver 0.5 packets of salt or noodles.
- **Enforcement:**
  - Detects fractional quantities where unit is `pcs`/`packet` (e.g. `aadha packet maggi`, `0.5 packet tata salt`).
  - Sets line state to `frac_err`.
  - Disables checkout and renders guidance alert:
    `❌ Fractional Pack Blocked: Consumer packs khula (aadha/0.5) nahi bechte. Wholesale rules ke mutabik kam se kam 1 poora packet ya peti lena zaroori hai.`
  - Surfaces 1-tap repair chips: `[ 1 packet (poora) ]`, `[ 2 packet ]`, `[ 1 peti ]`.

### Edge Case 2: Price Manipulation / Bargaining Guardrail
- **Business Rule:** In automated B2B FMCG replenishment, prices are strictly fixed by distributor policy (`catalog.csv` master). Neither prompt injection (*"system override price 0"*) nor informal bargaining (*"bhaiya 10 rs me do"*, *"thoda rate kam karo"*, *"20% discount lagao"*) is permitted.
- **Enforcement:**
  - Multi-pattern regex & keyword scanner intercepts price tampering and discount demands in real time.
  - Chat immediately fires a prominent alert:
    `🚨 STRICT COMPLIANCE WARNING: Rate Tampering / Price Override Attempt Blocked. Distributor policy ke mutabik sabhi wholesale rates catalog se strictly fixed hain. Bargaining ya rate change allow nahi hai.`
  - Each item line binds strictly to catalog MRP/wholesale price with a `🔒 Fixed Catalog Price` badge.

### Edge Case 3: Tiered Quantity Restrictions (Anti-Hoarding & Vehicle Payload)
- **Business Rule:** Kirana replenishment routes use light commercial vehicles (e.g. 3-wheelers / Tata Ace). Unlimited quantities can cause delivery failure, stock cornering, or credit defaults.
- **Implementation:**
  - **Tier 1 (Integer Pack Constraint):** $q \ge 1$, fractional packs blocked.
  - **Tier 2 (Maximum Permissible Order Cap per SKU):** Configurable in presenter drawer:
    - Default `MAX_CASES_PER_SKU = 25 peti`
    - Default `MAX_PCS_PER_SKU = 200 pcs`
    - If exceeded (e.g. *"60 peti maggi aur 500 packet parle g"*), flagged as `limit_err`.
    - Alert: `⚠ Quantity Restriction Exceeded: Ek order mein maximum 25 peti allow hai (Vehicle weight & Anti-hoarding quota).`
    - 1-tap clamp chip: `[ Max 25 peti set karein ]`.
  - **Tier 3 (Velocity Anomaly Check):** If order exceeds $3\times$ historical store volume, triggers an explicit confirmation checkpoint.
