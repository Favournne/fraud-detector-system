import json
import time
import redis
from confluent_kafka import Consumer

KAFKA_BROKERS = "127.0.0.1:9092"
INPUT_TOPIC = "transactions"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
user_buffers = {}

def update_user(user, amount, now):
    buf = user_buffers.setdefault(user, {'timestamps': [], 'amounts': []})
    buf['timestamps'].append(now)
    buf['amounts'].append(amount)
    cutoff = now - 3600
    while buf['timestamps'] and buf['timestamps'][0] < cutoff:
        buf['timestamps'].pop(0)
        buf['amounts'].pop(0)
    velocity_1m = 0
    sum_1h = 0.0
    cutoff_1m = now - 60
    for ts, amt in zip(buf['timestamps'], buf['amounts']):
        if ts >= cutoff_1m:
            velocity_1m += 1
        sum_1h += amt
    avg_1h = sum_1h / len(buf['amounts']) if buf['amounts'] else 0.0
    redis_client.setex(f"user:{user}", 3600, json.dumps({
        "velocity_1m": velocity_1m,
        "avg_amount_1h": round(avg_1h, 2)
    }))
    print(f"✅ Redis updated for {user}: velocity={velocity_1m}, avg_1h={avg_1h:.2f}")

consumer = Consumer({
    'bootstrap.servers': KAFKA_BROKERS,
    'group.id': 'aggregator',
    'auto.offset.reset': 'earliest',
})
consumer.subscribe([INPUT_TOPIC])

print("Aggregator running (keying by nameOrig)...")
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            continue
        tx = json.loads(msg.value().decode('utf-8'))
        user = tx.get('nameOrig')          # Changed from account_number
        if user is None:
            # fallback to account_number if nameOrig missing
            user = tx.get('account_number')
        amount = float(tx.get('amount', 0.0))
        if user:
            update_user(user, amount, time.time())
except KeyboardInterrupt:
    pass
finally:
    consumer.close()