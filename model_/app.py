import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import redis  # Hooking back into your real Windows/Docker port!

# Initialize Production Server Engine
app = FastAPI(
    title="Stateful Production Fraud Inference Engine",
    description="Enterprise-Grade Real-Time Redis State Machine & Gate Engine",
    version="2.0.0"
)

MODEL_PATH = r"C:\Users\USER\fraud_project\model_\models\fraud_pipeline_v1.0.pkl"
OPERATIONAL_THRESHOLD = 0.75  # Calibrated for industry-standard precision/recall balance

try:
    pipeline = joblib.load(MODEL_PATH)
    print("Production ML Pipeline successfully mapped into server memory!")
except Exception as e:
    print(f"Critical Error loading pipeline: {str(e)}")
    pipeline = None

# Connect directly to your live Docker container or native port running background operations
try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    # Ping verification to catch connection issues immediately before endpoint calls
    r.ping()
    print("Successfully connected to live local Redis instance on port 6379!")
except Exception as e:
    print(f"Connection Warning: Could not reach live Redis instance. Ensure container is active: {str(e)}")

# Audit logger datastore for internal customer care representative lookups
BLOCKED_ACCOUNTS_DB = {}


class TransactionPayload(BaseModel):
    transaction_id: str = Field(..., example="TXN-100293")
    account_number: str = Field(..., example="3019845210")
    customer_name: str = Field(..., example="Olumide Balogun")
    device_id: str = Field(..., example="DEV-99481X")
    amount: float = Field(..., example=450000.00)
    channel: str = Field(..., example="Mobile App")
    transaction_type: str = Field(..., example="Debit")
    destination_bank: str = Field(..., example="Opay")
    narration: str = Field(..., example="Business Invoice Payment")
    location_state: str = Field(..., example="Lagos")


# =====================================================================
# STATE MACHINE MECHANICS (Using Real Redis Pipelines)
# =====================================================================
def get_and_prune_redis_metrics(acct, device, current_amount, current_bank):
    """
    Leverages Redis transactional pipeline structures to fetch and update
    real-time rolling transaction metrics in a single network trip.
    """
    pipe = r.pipeline()
    clean_dev = device if (device and pd.notna(device)) else f"{acct}_unk_dev"
    
    # Generate keys
    user_tx_key = f"tx_count:1h:{acct}"
    dev_tx_key = f"dev_count:1h:{clean_dev}"
    user_bank_key = f"user_banks:1h:{acct}"
    dev_acct_key = f"dev_accts:24h:{clean_dev}"
    user_7d_key = f"user_amt:7d:{acct}"
    
    # Fetch current transaction velocity states
    pipe.get(user_tx_key)
    pipe.get(dev_tx_key)
    pipe.smembers(user_bank_key)
    pipe.smembers(dev_acct_key)
    pipe.lrange(user_7d_key, 0, -1)
    
    # Execute lookups
    res_user_tx, res_dev_tx, res_banks, res_accts, res_amounts = pipe.execute()
    
    # Parse lookups into baseline floats for the model schema
    user_tx_count_1h = float(res_user_tx) + 1.0 if res_user_tx else 1.0
    device_tx_count_1h = float(res_dev_tx) + 1.0 if res_dev_tx else 1.0
    
    unique_banks = set(res_banks) if res_banks else set()
    unique_banks.add(current_bank)
    unique_dest_banks_1h = float(len(unique_banks))
    
    unique_accounts = set(res_accts) if res_accts else set()
    unique_accounts.add(acct)
    accounts_per_device_24h = float(len(unique_accounts))
    
    if not res_amounts:
        amount_vs_avg_7d = 1.0
    else:
        history_floats = [float(x) for x in res_amounts]
        avg_7d = np.mean(history_floats)
        amount_vs_avg_7d = float(current_amount / avg_7d) if avg_7d > 0 else 1.0
        
    return user_tx_count_1h, device_tx_count_1h, unique_dest_banks_1h, accounts_per_device_24h, amount_vs_avg_7d


def commit_approved_transaction_to_redis(acct, device, current_amount, current_bank):
    """Updates global keys only after the inference engine approves a transaction."""
    pipe = r.pipeline()
    clean_dev = device if (device and pd.notna(device)) else f"{acct}_unk_dev"
    
    user_tx_key = f"tx_count:1h:{acct}"
    dev_tx_key = f"dev_count:1h:{clean_dev}"
    user_bank_key = f"user_banks:1h:{acct}"
    dev_acct_key = f"dev_accts:24h:{clean_dev}"
    user_7d_key = f"user_amt:7d:{acct}"
    
    # Atomic transaction counter increments
    pipe.incr(user_tx_key)
    pipe.expire(user_tx_key, 3600)
    
    pipe.incr(dev_tx_key)
    pipe.expire(dev_tx_key, 3600)
    
    # Add items to tracking sets
    pipe.sadd(user_bank_key, current_bank)
    pipe.expire(user_bank_key, 3600)
    
    pipe.sadd(dev_acct_key, acct)
    pipe.expire(dev_acct_key, 86400)
    
    # Maintain sliding historical lists
    pipe.lpush(user_7d_key, current_amount)
    pipe.ltrim(user_7d_key, 0, 99)  # Caps memory ceiling limits to 100 entries per user
    pipe.expire(user_7d_key, 604800)
    
    pipe.execute()


# =====================================================================
# INFERENCE ENFORCEMENT POST ENDPOINT
# =====================================================================
@app.post("/v1/predict")
async def predict_transaction(payload: TransactionPayload):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Inference engine offline.")
    
    try:
        now = datetime.now()
        raw_data = payload.dict()
        df_input = pd.DataFrame([raw_data])
        df_input['timestamp'] = pd.to_datetime(now)
        
        # Pull live operational indicators straight out of our active cache container
        u1h, d1h, b1h, dev24h, avg7d = get_and_prune_redis_metrics(
            payload.account_number, payload.device_id, payload.amount, payload.destination_bank
        )
        
        # Formulate feature matrix layout
        df_input['user_tx_count_1h'] = u1h
        df_input['device_tx_count_1h'] = d1h
        df_input['unique_dest_banks_1h'] = b1h
        df_input['accounts_per_device_24h'] = dev24h
        df_input['amount_vs_avg_7d'] = avg7d
        
        X_input = df_input.drop(columns=['is_fraud', 'transaction_id', 'customer_name', 'narration'], errors='ignore')
        
        # Calculate risk scores
        fraud_probability = float(pipeline.predict_proba(X_input)[0, 1])
        action = "BLOCKED" if fraud_probability >= OPERATIONAL_THRESHOLD else "APPROVED"
        
        if action == "BLOCKED":
            # Cache full diagnostic logs for internal security agents
            BLOCKED_ACCOUNTS_DB[payload.account_number] = {
                "transaction_id": payload.transaction_id,
                "customer_name": payload.customer_name,
                "flagged_amount": payload.amount,
                "fraud_probability_score": round(fraud_probability, 4),
                "blocked_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "live_behavioral_signals": {
                    "user_tx_count_last_hour": u1h,
                    "device_tx_count_last_hour": d1h,
                    "distinct_banks_targeted_1h": b1h,
                    "accounts_switched_on_device_24h": dev24h,
                    "amount_deviation_multiplier_7d": round(avg7d, 2)
                }
            }
        else:
            # Commit mutations to cache only if transaction clears the risk gate
            commit_approved_transaction_to_redis(
                payload.account_number, payload.device_id, payload.amount, payload.destination_bank
            )
            
        return {
            "transaction_id": payload.transaction_id,
            "action": action,
            "public_message": "APPROVED" if action == "APPROVED" else "Transaction Declined. Please contact customer support."
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"State Engine Error: {str(e)}")


@app.get("/v1/accounts/{account_number}/fraud-telemetry")
async def get_account_telemetry(account_number: str):
    if account_number not in BLOCKED_ACCOUNTS_DB:
        raise HTTPException(status_code=404, detail="No active security blocks on this account profile.")
    return {"account_number": account_number, "status": "RESTRICTED", "telemetry": BLOCKED_ACCOUNTS_DB[account_number]}