Real-Time Fraud Detection System
A stateful, real‑time fraud detection pipeline powered by Kafka, Redis, PostgreSQL, FastAPI, and XGBoost.
It scores transactions using account‑level context (transaction velocity and average amount over the last simulated hour) to catch fraud patterns that isolated transactions would miss.

Architecture
Producer (backend/producer_live.py) – Simulates a live transaction stream by sampling from the PaySim dataset, injecting noise, and publishing to Kafka (transactions).

Aggregator (backend/aggregator.py) – Consumes Kafka, computes tx_count_recent and avg_amount_recent per account over a sliding hour‑window, and writes to Redis.

Scoring – Two paths:

Kafka Consumer (backend/consumer_predict_confluent.py) – scores every transaction automatically.

FastAPI (backend/main.py) – REST API for on‑demand scoring.

PostgreSQL – stores all predictions for dashboard KPIs and account lookup.

Dashboards – Two Streamlit apps:

Live Monitor (port 8501) – shows KPIs and real‑time transaction table (BLOCKED rows in red).

Investigation Portal (port 8502) – search accounts, view history, and read SHAP‑based explanations.

Tech Stack
Infrastructure: Docker (Kafka, Redis, PostgreSQL)

Streaming: Apache Kafka, confluent-kafka

Features: Redis (low‑latency cache)

Storage: PostgreSQL

ML: XGBoost, scikit‑learn

API: FastAPI

Monitoring: Streamlit

Prerequisites
Docker Desktop

Python 3.10+ (virtual environment recommended)

Setup
bash
# 1. Clone the repo
git clone https://github.com/Favournne/fraud-detector-system.git
cd fraud_project

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start infrastructure (PostgreSQL, Redis, Kafka)
docker-compose up -d

# 5. Create Kafka topics (optional – the producer/consumer will auto‑create them)
docker exec kafka kafka-topics --create --topic transactions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --create --topic fraud_results --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

Running the Pipeline
Open six terminals for the full system:

Terminal	Command
1	python backend/aggregator.py
2	python backend/producer_live.py
3	python backend/consumer_predict_confluent.py
4	uvicorn backend.main:main --host 0.0.0.0 --port 8000 --reload
5	streamlit run frontend/dashboard.py --server.port 8501
6	streamlit run frontend/get.py --server.port 8502
Alternatively, use the provided start_all.ps1 (Windows) to launch everything with a single command.

Retraining the Model
bash
cd retrain
python retrain_model.py
# Then restart FastAPI and the Kafka consumer to load the new model.
The retraining script:

Uses the existing splits/train.csv, validation.csv, test.csv

Computes rolling features over a single‑hour window (PaySim's step granularity)

Selects the threshold on the validation set using a 50% precision‑floor rule

Saves the new model, scaler, and threshold to data_/

API Endpoints
Method	Endpoint	Description
POST	/v1/predict	Score a single transaction
GET	/v1/health	Health check
GET	/v1/account/{nameOrig}/transactions	Last 10 transactions with SHAP explanations
GET	/v1/account/{nameOrig}/stats	Summary statistics (total, blocked, total amount, avg amount)
GET	/v1/account/{nameOrig}/last	Most recent transaction with explanation
Sample POST payload:

json
{
  "transaction_id": "TXN-123",
  "amount": 5000.00,
  "oldbalanceOrg": 10000.00,
  "oldbalanceDest": 2000.00,
  "step": 10,
  "transaction_type": "TRANSFER",
  "nameOrig": "C123456789"
}

Model Performance (Final)
Metric	Value
AUC‑ROC	0.9987
Precision	51.9%
Recall	88.0%
F1‑Score	0.65
Threshold	0.0525
Test set size	415,562 transactions (1,232 fraud cases)
Confusion Matrix (at threshold 0.0525):

Predicted Legit	Predicted Fraud
Actual Legit	413,326	1,004
Actual Fraud	148	1,084
The threshold was selected on the validation set with a 50% precision floor and generalised well to the test set (51.9% precision).

Project Structure
text
fraud_project/
├── backend/
│   ├── aggregator.py
│   ├── producer_live.py
│   ├── consumer_predict_confluent.py
│   ├── main.py                     # FastAPI application
│   └── __init__.py
├── frontend/
│   ├── dashboard.py                # Live Monitor (port 8501)
│   └── get.py                      # Investigation Portal (port 8502)
├── retrain/
│   └── retrain_model.py            # Retraining pipeline
├── data_/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   ├── threshold.json
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   └── splits/                     # train/validation/test CSV splits
├── docker-compose.yml
├── .env                            # Environment variables (not committed)
├── .gitignore
├── requirements.txt
├── start_all.ps1                   # One‑click launcher for Windows
└── README.md
Notes on the Corrections
The system has undergone a thorough reconciliation pass. Key fixes include:

Feature windowing: Both behavioural features are now computed over the last simulated hour (step - 1), reflecting PaySim's hour‑granular data.

Threshold: Re‑derived on the validation set using a 50% precision floor, yielding 0.0525.

Splits: Training, validation, and test sets are non‑overlapping, verified via row‑level checks.

Investigation Portal: Now persists tx_count_recent and avg_amount_recent in PostgreSQL, so SHAP explanations use real behavioural values.

Acknowledgments
Built as a portfolio project to demonstrate real‑time ML engineering skills. The PaySim dataset is used for simulation; the system is designed to be adaptable to real transaction feeds.

