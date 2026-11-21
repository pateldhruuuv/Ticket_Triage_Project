# Support Ticket Triage Service

An intelligent, production-ready AI/ML support ticket triage service that cleans incoming customer tickets, classifies them into **Categories** and **Priorities**, estimates prediction confidence, and automatically routes uncertain or degraded tickets to a human review queue.

---

## 1. Design Note: Scaling to 50,000 Tickets/Day & 30-Day Observability

> **Deliverable 1 for Leadership & Engineering Review**

To scale this ticket triage system to handle **50,000 tickets/day (~0.6–2 tickets/second average, peaking at 15–25 tickets/second)** with near real-time SLAs and ensure it remains robust and accurate 30 days post-launch:

* **Asynchronous Event-Driven Architecture & Scalable Workers**: Decouple ticket ingestion from classification using a distributed message queue (e.g., Apache Kafka or AWS SQS). Ingestion webhooks push tickets immediately into an `incoming-tickets` topic. A fleet of containerized Python workers (deployed on Kubernetes with Horizontal Pod Autoscalers based on queue depth) consumes batches of tickets. Batch inference enables sub-50ms latency per ticket and scales compute elastically during peak hours.
* **SLA-Partitioned Priority Routing & Graceful Degradation**: Workers publish triaged tickets to partitioned output queues (`priority-urgent`, `priority-high`, `normal`, and `human-review`). A circuit-breaker pattern ensures that if model inference latency degrades or an external model dependency fails, the system automatically falls back to lightweight deterministic rule heuristics, flagging all borderline tickets for review rather than dropping tickets.
* **Semantic Caching & Deduplication Layer**: Deploy an in-memory Redis cluster caching embeddings and hashes of frequent, repetitive customer inquiries (e.g., promo code questions, standard invoice requests). A semantic similarity threshold (cosine similarity > 0.96) against recently triaged tickets enables immediate zero-compute classification, bypassing full inference for ~20–30% of high-volume inquiries.
* **Continuous Drift Detection & Distribution Monitoring**: Ingest all production inferences and feature vectors into an observability engine (e.g., Prometheus + Grafana with Evidently AI). Track:
  1. **Data Drift**: Shifts in vocabulary, ticket length, or non-English language proportions (monitored via Population Stability Index / Wasserstein distance).
  2. **Prediction Drift**: Abrupt shifts in category or priority distributions (e.g., sudden spike in "Urgent" or "Billing" indicating an active billing outage).
  3. **Confidence Decay**: Rolling average of $C_{\text{overall}}$ and margin scores; a downward drift triggers automated alerts.
* **Human-in-the-Loop Active Learning & Review Dispute Tracking**: The 10–15% of tickets routed to human review serve as an active learning stream. When human agents verify or reclassify a ticket, the correction is logged as ground truth. We track the **Reviewer Dispute Rate** (how often the agent overrides the model’s tentative prediction). If dispute rates exceed 5% in any category, an alert fires for model recalibration.
* **Automated Weekly Retraining & Shadow / Canary Deployment**: High-confidence corrections from human reviewers are automatically validated and merged into a curated gold evaluation dataset. A weekly CI/CD pipeline retrains and evaluates candidate model checkpoints against the benchmark. New candidate models run in **Shadow Mode** (evaluating live production traffic alongside the active model without routing) for 48 hours before progressive canary rollout (10% → 50% → 100%).

---

## 2. Project Architecture & Flow

### Directory Structure
```
support-ticket-triage/
│
├── data/
│   └── tickets.csv          # 300-ticket raw customer dataset
│
├── output/
│   └── results.csv          # Final triaged output (id, category, priority, needs_review)
│
├── src/
│   ├── __init__.py          # Package initializer
│   ├── preprocessing.py     # Text cleaning, boilerplate stripping, missing value handling
│   ├── classifier.py        # Multi-lingual semantic anchors, rule boosts & priority heuristics
│   ├── triage.py            # Confidence evaluation and automated vs. human routing
│   └── utils.py             # Validation, logging, formatting, and reporting
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Automated unit tests covering edge cases & components
│
├── main.py                  # End-to-end CLI orchestrator
├── requirements.txt         # Project dependencies
├── README.md                # Architecture, documentation, and design note
└── .gitignore               # Git ignore configuration
```

### Pipeline Flow Diagram
```
              tickets.csv
                   │
                   ▼
                main.py
                   │
                   ▼
           preprocessing.py
                   │
                   │ Clean subject + body
                   │ Impute NaN values
                   │ Detect degraded / gibberish text
                   ▼
            classifier.py
                   │
                   │ Category (Billing, Technical, Account, Shipping, Other)
                   │ Priority (Low, Medium, High, Urgent)
                   │ Confidence Scores & Margins
                   ▼
               triage.py
                   │
       ┌───────────┴───────────┐
       │                       │
   Confident               Uncertain
(Conf >= 0.65)          (Conf < 0.65 or Degraded)
       │                       │
       ▼                       ▼
Category + Priority       Human Review
(needs_review="no")    (needs_review="yes")
       │                       │
       └───────────┬───────────┘
                   ▼
              results.csv
```

---

## 3. Component Details & Methodology

### 3.1 Preprocessing (`src/preprocessing.py`)
- **Missing Value Imputation**: Detects empty or null text (e.g. ticket `T-0093` with NaN body) and standardizes missing values.
- **Boilerplate Stripping**: Removes common email salutations ("Hello support,", "Hi team,", "Good morning,") and sign-offs ("Kind regards,", "Thanks in advance,", "Sent from my phone") to isolate the core customer intent.
- **Degraded Text & Spam Detection**: Automatically flags tickets that contain gibberish (e.g. `T-0261`: `??? / asdf asdf test test 123`), single-word vague cries (`T-0007`: `help / help`), or empty content, routing them directly to human review.

### 3.2 Classification Engine (`src/classifier.py`)
- **K-Means Clustering on TF-IDF Vectors**:
  - **Category Vectorizer & Model**: Fits `TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)` and `KMeans(n_clusters=5, random_state=42, n_init=10)`.
  - **Priority Vectorizer & Model**: Fits `TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)` and `KMeans(n_clusters=4, random_state=42, n_init=10)`.
  - **Centroid Term Inspection**: Retrieves top cluster terms via `get_category_cluster_terms()` and `get_priority_cluster_terms()`.
  - **Cluster Distance Evaluation**: Calculates distance of each ticket to its cluster center via `get_category_distances()` and `get_priority_distances()`.
  - **Soft Probability & Margin Calibration**: Computes soft K-Means probabilities over cluster distances using temperature-scaled exponential mapping:
    $$P(k|x) = \frac{e^{-\beta (d_k - d_{\min})}}{\sum_{j=1}^K e^{-\beta (d_j - d_{\min})}}$$
  - **Category & Urgency Mapping**: Maps cluster centroids and fine-grained patterns to required labels (`Billing`, `Technical`, `Account`, `Shipping`, `Other` and `Low`, `Medium`, `High`, `Urgent`). Escalates fraud and compromised accounts to `Urgent`.

### 3.3 Triage Decision Engine (`src/triage.py`)
- Evaluates overall confidence against configurable threshold (default `0.50`).
- Enforces strict safety rails:
  - If text is degraded or empty -> `needs_review = "yes"`.
  - If category margin is too narrow (ambiguous multi-cluster boundary) -> `needs_review = "yes"`.
  - If category or priority confidence is below minimum safety bounds -> `needs_review = "yes"`.
  - For tickets routed to human review, `category` and `priority` are left empty (`""`) per specification, or preserved if `--preserve-tentative` is enabled.

---

## 4. Setup & Execution

### Prerequisites
- Python 3.10+
- Dependencies: `pandas`, `numpy`, `scikit-learn`

### Installation
```bash
# Clone or navigate to the repository
cd support-ticket-triage

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# Run with default settings (reads data/tickets.csv, outputs output/results.csv)
python main.py

# Custom threshold or file paths
python main.py --input data/tickets.csv --output output/results.csv --threshold 0.50

# Optionally preserve tentative AI predictions for human reviewers
python main.py --preserve-tentative
```

### Running Unit Tests
```bash
python -m unittest discover tests -v
```

---

## 5. Execution Results on 300 Tickets Benchmark

```
=================================================================
           SUPPORT TICKET TRIAGE EXECUTION REPORT
=================================================================
 Total Tickets Processed : 300
 Auto-Triaged (Confident):  246 ( 82.0%)
 Sent to Human Review    :   54 ( 18.0%)
-----------------------------------------------------------------
 Breakdown by Predicted Category (Auto-Triaged):
   - Billing     : 101 ( 41.1%)  ##########
   - Technical   :  45 ( 18.3%)  ####
   - Account     :  51 ( 20.7%)  #####
   - Shipping    :  17 (  6.9%)  #
   - Other       :  32 ( 13.0%)  ###
-----------------------------------------------------------------
 Breakdown by Priority (Auto-Triaged):
   - Urgent      :  16 (  6.5%)  #
   - High        :  35 ( 14.2%)  ###
   - Medium      : 139 ( 56.5%)  ##############
   - Low         :  56 ( 22.8%)  #####
-----------------------------------------------------------------
 Reasons for Escalation to Human Review:
   - Degraded text                   :   3
   - Low category confidence         :  49
   - Ambiguous category margin       :   1
   - Overall confidence below threshold:   1
=================================================================
```

### Key Spot-Check Validations
| Ticket ID | Snippet / Scenario | Output Category | Output Priority | Needs Review | Note |
|---|---|---|---|---|---|
| `T-0001` | Order LM-520106 arrived with EU plug | Shipping | High | `no` | Wrong hardware item delivered |
| `T-0004` | Invoice copy for expense report | Billing | Low | `no` | Standard informational billing query |
| `T-0007` | Subject: "help", Body: "help" | *(empty)* | *(empty)* | `yes` | Vague single-word Cry for help |
| `T-0008` | Someone is using my saved card | Billing | Urgent | `no` | Active fraud / unauthorized transaction |
| `T-0017` | I think my account was hacked | Account | Urgent | `no` | Security takeover / compromise |
| `T-0020` | Phone stolen with authenticator app | Account | Urgent | `no` | 2FA lockout emergency |
| `T-0032` | Rechnung benötigt (German invoice) | Billing | Low | `no` | Multi-lingual invoice request |
| `T-0037` | मेरे कार्ड से पैसे कट रहे हैं (Hindi fraud) | Billing | Urgent | `no` | Multi-lingual unauthorized card deduction |
| `T-0039` | アプリが遅い (Japanese app lag) | Technical | Medium | `no` | Multi-lingual performance bug |
| `T-0093` | (no subject), Body: NaN | *(empty)* | *(empty)* | `yes` | Missing text input |
| `T-0261` | ??? / asdf asdf test test 123 | *(empty)* | *(empty)* | `yes` | Gibberish / placeholder test |
