# Fraud ML + Kafka Workspace

This project scaffolds a Python ML pipeline that generates synthetic fraud transaction data with `Faker`, streams it through Kafka, and trains a simple fraud detection model.

## What is included

- `docker-compose.yml` for Kafka + Zookeeper
- `requirements.txt` with Kafka, Faker, ML, and data libraries
- `src/data_generator.py` to publish fake transaction events
- `src/consumer.py` to consume Kafka events into a CSV file
- `src/model/trainer.py` to train a fraud model from stored transaction data
- `.env.example` for Kafka configuration

## Setup

1. Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Start Kafka

```powershell
docker compose up -d
```

4. Copy environment variables

```powershell
copy .env.example .env
```

5. Produce fake Kafka events

```powershell
python -m src.data_generator --count 200
```

6. Run the consumer to save events to `data/transactions.csv`

```powershell
python -m src.consumer --count 200
```

7. Train the model

```powershell
python -m src.model.trainer --input data/transactions.csv --output model/fraud_model.joblib
```

## Notes

- `topic` defaults to `fraud-transactions`
- Kafka broker defaults to `localhost:9092`
- The training script uses a simple `RandomForestClassifier` for demo purposes
