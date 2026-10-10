
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

NUM_TRANSACTIONS = 5000

categories = {
    "Food": ["Swiggy", "Zomato", "Cafe"],
    "Shopping": ["Amazon", "Flipkart", "Myntra"],
    "Transport": ["Uber", "Ola", "Metro"],
    "Groceries": ["Blinkit", "Zepto", "BigBasket"],
    "Entertainment": ["Netflix", "PVR", "Spotify"],
    "Bills": ["Electricity", "Mobile Recharge", "Internet"],
    "Education": ["Books", "Online Courses", "Stationery"],
    "Healthcare": ["Pharmacy", "Hospital", "Clinic"],
    "Travel": ["IRCTC", "MakeMyTrip", "Hotel"],
}

category_names = list(categories.keys())

category_probs = [
    0.22, 0.16, 0.14, 0.15, 0.08,
    0.10, 0.06, 0.05, 0.04
]

start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2026-09-30")

days = (end_date - start_date).days + 1

records = []

for i in range(NUM_TRANSACTIONS):
    category = rng.choice(category_names, p=category_probs)
    merchant = rng.choice(categories[category])

    date = start_date + pd.Timedelta(
        days=int(rng.integers(0, days))
    )

    hour = int(rng.integers(7, 24))
    minute = int(rng.integers(0, 60))

    timestamp = date + pd.Timedelta(
        hours=hour, minutes=minute
    )

    amount = round(
        float(rng.lognormal(mean=5.7, sigma=0.9)), 2
    )

    if category in ["Shopping", "Travel", "Healthcare"]:
        amount = round(amount * 2.5, 2)

    if rng.random() < 0.02:
        amount = round(amount * 8, 2)

    transaction_type = rng.choice(
        ["Debit", "Credit"], p=[0.88, 0.12]
    )

    status = rng.choice(
        ["Success", "Failed", "Pending"],
        p=[0.94, 0.04, 0.02]
    )

    payment_method = rng.choice(
        ["Google Pay", "PhonePe", "Paytm", "BHIM"],
        p=[0.35, 0.35, 0.20, 0.10]
    )

    records.append({
        "transaction_id": f"UPI{i+1:06d}",
        "date": timestamp.date().isoformat(),
        "time": timestamp.strftime("%H:%M:%S"),
        "amount": amount,
        "transaction_type": transaction_type,
        "category": category,
        "merchant": merchant,
        "payment_method": payment_method,
        "status": status,
        "day_of_week": timestamp.strftime("%A"),
        "month": timestamp.strftime("%B"),
        "year": timestamp.year
    })

df = pd.DataFrame(records)
df = df.sort_values(["date", "time"]).reset_index(drop=True)

output_path = Path(__file__).resolve().parent.parent / "data"
output_path.mkdir(parents=True, exist_ok=True)

csv_path = output_path / "upi_transactions.csv"
df.to_csv(csv_path, index=False)

print("Dataset generated successfully!")
print("Total transactions:", len(df))
print("Dataset saved at:", csv_path)
print(df.head())
