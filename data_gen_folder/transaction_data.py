import csv
import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
from kafka import KafkaProducer

class NigerianTransactionSimulator:
    """
    Simulates realistic financial transactions for the Nigerian banking ecosystem.
    Implements stateful user profiles and time-series syndicated fraud patterns.
    """
    
    def __init__(self, num_customers: int = 1000, locale: str = 'en_NG'):
        self.fake = Faker([locale])
        self.banks = ["Access Bank", "GTBank", "Zenith Bank", "UBA", "First Bank", "Opay", "Palmpay", "Moniepoint"]
        self.channels = ["POS", "ATM", "Mobile App", "USSD Code (*737#)", "Web (Paystack)"]
        self.narrations = ["Funds Transfer", "Pocket Money", "Payment for goods", "Data top-up", "Family support", "Urgent Outward Transfer"]
        
        self.states = [
            "Lagos", "Abuja (FCT)", "Rivers (PH)", "Kano", "Oyo (Ibadan)", 
            "Kaduna", "Anambra", "Delta", "Enugu", "Ondo", "Edo", "Ogun"
        ]
        self.state_weights = [0.35, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05, 0.04, 0.04, 0.03, 0.02, 0.02]
        
        self.customer_profiles = self._generate_customer_profiles(num_customers)
        self.active_rogue_devices = [f"DEV-ROGUE{str(i).zfill(2)}" for i in range(15)]

        try:
            self.producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.topic_name = 'raw_transactions'
            print(f"Kafka Producer successfully attached to broker topic: '{self.topic_name}'")
        except Exception as e:
            print(f"Kafka Connection Error: {e}")
            self.producer = None

    def _generate_customer_profiles(self, num_customers: int) -> dict:
        profiles = {}
        for _ in range(num_customers):
            acct_num = f"{random.randint(1, 9)}{random.randint(100000000, 999999999)}"
            segment_roll = random.random()

            if segment_roll < 0.70:
                avg_amount = random.uniform(1500, 15000)
            elif segment_roll < 0.95:
                avg_amount = random.uniform(15001, 80000)
            else:
                avg_amount = random.uniform(80001, 350000)

            profiles[acct_num] = {
                "name": self.fake.name(),
                "home_state": random.choices(self.states, weights=self.state_weights, k=1)[0],
                "avg_amount": round(avg_amount, 2),
                "active_hours": list(range(6, 23)),
                "primary_device": f"DEV-{random.randint(100000, 999999)}"
            }
        return profiles

    def _generate_amount(self, profile: dict) -> float:
        if random.random() < 0.95:
            return round(random.uniform(profile["avg_amount"] * 0.2, profile["avg_amount"] * 2.0), 2)
        return round(random.uniform(200, profile["avg_amount"] * 5.0), 2)

    def _evaluate_fraud_rules(self, profile: dict, channel: str, amount: float, state: str, tx_time: datetime, is_known_device: bool) -> int:
        """Evaluates baseline behavioral anomalies with realistic, subtler thresholds."""
        is_night = tx_time.hour in [0, 1, 2, 3, 4]
        is_evening = tx_time.hour in [20, 21, 22, 23]
        
        # Scenario A: USSD drain — lowered from >100k fixed to >2x avg, evening OR night
        if channel == "USSD Code (*737#)" and amount > (profile["avg_amount"] * 2) and (is_night or is_evening):
            return 1 if random.random() < 0.60 else 0  
        
        # Scenario B: POS anomaly — Fixed indentation so sum() stays local to POS context
        if channel == "POS":
            amount_spike = amount > (profile["avg_amount"] * 4)
            time_unusual = tx_time.hour not in profile["active_hours"]
            location_changed = state != profile["home_state"]
            if sum([amount_spike, time_unusual, location_changed]) >= 1:
                return 1 if random.random() < 0.45 else 0
        
        # Scenario C: Unknown device — Fixed indentation so sum() stays local to Digital context
        if channel in ["Mobile App", "Web (Paystack)"] and not is_known_device:
            amount_spike = amount > (profile["avg_amount"] * 3)
            time_unusual = tx_time.hour not in profile["active_hours"]
            if sum([amount_spike, time_unusual]) >= 1:
                return 1 if random.random() < 0.70 else 0
        
        # Scenario D: Large ATM withdrawal outside active hours
        if channel == "ATM" and amount > (profile["avg_amount"] * 5) and tx_time.hour not in profile["active_hours"]:
            return 1 if random.random() < 0.50 else 0
        
        return 0

    def generate_dataset(self, filename: str, num_records: int, stream_live: bool = True):
        print(f"Manufacturing {num_records} highly realistic behavioral records...")

        current_time = datetime.now()
        account_list = list(self.customer_profiles.keys())

        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = [
                "transaction_id", "timestamp", "account_number", "customer_name",
                "device_id", "amount", "channel", "transaction_type", "destination_bank",
                "narration", "location_state", "is_fraud"
            ]
            writer.writerow(header)

            for i in range(num_records):
                # Tick time forward sequentially
                current_time += timedelta(seconds=random.uniform(1, 5))

                account_number = random.choice(account_list)
                profile = self.customer_profiles[account_number]

                channel = random.choice(self.channels)
                amount = self._generate_amount(profile)
                state = profile["home_state"] if random.random() < 0.92 else random.choices(self.states, weights=self.state_weights, k=1)[0]
                tx_type = "Debit" if random.random() > 0.25 else "Credit"

                # Assign device configuration
                if channel in ["POS", "ATM", "USSD Code (*737#)"]:
                    device_id = ""
                    is_known_device = True
                else:
                    if random.random() < 0.97:
                        device_id = profile["primary_device"]
                        is_known_device = True
                    else:
                        device_id = f"DEV-{random.randint(100000, 999999)}"
                        is_known_device = False

                # Evaluate baseline behavioral anomalies
                is_fraud = self._evaluate_fraud_rules(profile, channel, amount, state, current_time, is_known_device)

                # --- INJECTION PATTERN 1: Syndicated Device Pooling Attack ---
                if i > 1000 and i % 211 == 0:
                    is_fraud = 1
                    channel = "Mobile App"
                    device_id = random.choice(self.active_rogue_devices)
                    amount = round(random.uniform(profile["avg_amount"] * 1.5, profile["avg_amount"] * 4), 2)
                    tx_type = "Debit" if random.random() < 0.85 else "Credit"

                # --- INJECTION PATTERN 2: High Out-Degree Bank Layering ---
                if i > 5000 and i % 401 == 0:
                    is_fraud = 1
                    channel = random.choice(["Mobile App", "Web (Paystack)"])
                    amount = round(random.uniform(profile["avg_amount"] * 2, profile["avg_amount"] * 6), 2)
                    tx_type = "Debit" if random.random() < 0.85 else "Credit"
                    state = random.choices(self.states, weights=self.state_weights, k=1)[0]
                
                narration_str = random.choice(self.narrations)
                tx_id = f"TXN-{random.randint(1000000000, 9999999999)}"

                record = [
                    tx_id,
                    current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    account_number,
                    profile["name"],
                    device_id if device_id != "" else "NaN",
                    amount,
                    channel,
                    tx_type,
                    random.choice(self.banks) if tx_type == "Debit" else "N/A",
                    narration_str,
                    state,
                    is_fraud
                ]
                writer.writerow(record)
                
                api_compatible_payload = {
                    "TransactionID": tx_id,
                    "AccountID": f"ACC_{account_number}",
                    "TransactionAmount": float(amount),
                    "TransactionDate": current_time.strftime("%Y-%m-%d"),
                    "TransactionType": str(tx_type),
                    "Location": str(state),
                    "DeviceID": str(device_id) if device_id != "" else "NaN",
                    "IP Address": self.fake.ipv4(),
                    "MerchantID": f"MERCH_{random.randint(100, 999)}",
                    "Channel": str(channel),
                    "TransactionDuration": round(random.uniform(10.0, 180.0), 1),
                    "LoginAttempts": random.randint(4, 5) if is_fraud == 1 else random.randint(1, 2),
                    "AccountBalance": round(random.uniform(5000, 500000), 2)
                }

                if stream_live and self.producer:
                    self.producer.send(self.topic_name, value=api_compatible_payload)
                    print(f"Broadcast [Record {i+1}]: {tx_id} | {channel} | ₦{amount:,} | Fraud Target Status: {is_fraud}")
                    time.sleep(1.5)

        print("Streaming session complete! Archive generated safely.")


if __name__ == "__main__":
    simulator = NigerianTransactionSimulator(num_customers=1000)
    simulator.generate_dataset("nigerian_bank_transactions_live.csv", num_records=2000, stream_live=True)