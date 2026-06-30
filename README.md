🚨 Real-Time Fraud Detection System

A stateful, real‑time fraud detection pipeline powered by Kafka, Redis, PostgreSQL, FastAPI, and XGBoost.  
It scores transactions using **account‑level context** (transaction velocity and average amount) to catch fraud patterns that isolated transactions would miss.

🏗️ Architecture

1. Producer (`producer_live.py`) – Streams synthetic transactions to Kafka (`transactions`).
2. Aggregator (`aggregator.py`) – Consumes Kafka, computes `velocity_1m` and `avg_amount_1h` per account, writes to **Redis**.
3. Scoring – Two ways:
    -Kafka Consumer (`consumer_predict_confluent.py`)– scores every Kafka transaction automatically.
    -FastAPI (`backend/main.py`) – REST API for on‑demand scoring.
4.  PostgreSQL – stores all predictions for dashboard KPIs and account lookup.
5.  Dashboard (`dashboard.py`) – Streamlit app showing KPIs (Total, Frauds, Money Saved, Fraud Rate) and a transaction table (BLOCKED rows in red).

🛠️ Tech Stack

- **Infrastructure**: Docker (Kafka, Redis, PostgreSQL)
- **Streaming**: Apache Kafka, `confluent-kafka`
- **Features**: Redis (low‑latency cache)
- **Storage**: PostgreSQL
- **ML**: XGBoost, scikit‑learn
- **API**: FastAPI
- **Monitoring**: Streamlit

---📋 Prerequisites

- Docker Desktop
- Python 3.10+ (virtual environment recommended)

---

⚙️ Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd fraud_project

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start infrastructure
docker-compose up -d

# 5. Create Kafka topics
docker exec kafka kafka-topics --create --topic transactions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --create --topic fraud_results --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

🚀 Running the Pipeline
Open 5 terminals:
Terminal
Command
1
python aggregator.py
2
python producer_live.py
3
python consumer_predict_confluent.py
4
uvicorn backend.main:main --host 0.0.0.0 --port 8000 --reload
5
streamlit run dashboard.py


🔬 Retraining the Model
bash
cd retrain
python retrain_model.py
# Then restart FastAPI & consumer to load the new model

📡 API Endpoints
Method
Endpoint
Description
POST
/v1/predict
Score a transaction
GET
/v1/health
Health check
GET
/v1/account/{nameOrig}/transactions
Last 10 transactions
GET
/v1/account/{nameOrig}/stats
Summary stats
GET
/v1/account/{nameOrig}/last
Most recent transaction

Sample POST:
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

📊 Model Performance (Latest)
Precision: 88.9%
Recall: 77.7%
F1 Score: 0.8288
False Positive Rate: ~0.03%
Blocking threshold: 0.3838

🗂️ Project Structure
text
fraud_project/
├── backend/
│   └── main.py
├── data_/
│   ├── paysim.csv
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── threshold.json
├── retrain/
│   └── retrain_model.py
├── aggregator.py
├── producer_live.py
├── consumer_predict_confluent.py
├── dashboard.py
├── docker-compose.yml
└── README.md

