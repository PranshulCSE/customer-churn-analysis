"""
generate_data.py

Yeh script ek realistic telecom customer dataset banata hai (CSV format).
Real world mein hum yeh data company ke CRM/billing system se lete hain,
yahan demo/practice ke liye synthetic data generate kar rahe hain so that
the project can run end-to-end without depending on an external download.

Run: python generate_data.py
Output: customer_churn.csv (same folder)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # total customers

# Tenure (months a customer has stayed with the company)
tenure_months = np.random.randint(1, 72, size=N)

# Monthly charges - depends loosely on a base plan + random variation
base_plan = np.random.choice([300, 500, 800, 1200], size=N, p=[0.35, 0.3, 0.25, 0.1])
monthly_charges = base_plan + np.random.normal(0, 50, size=N)
monthly_charges = np.clip(monthly_charges, 100, None).round(2)

# Total charges roughly = tenure * monthly_charges with some noise
total_charges = (tenure_months * monthly_charges) + np.random.normal(0, 200, size=N)
total_charges = np.clip(total_charges, 0, None).round(2)

# Number of support calls - unhappy customers usually call more
support_calls = np.random.poisson(2, size=N)

# Contract type
contract_type = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], size=N, p=[0.5, 0.3, 0.2])

# Internet service
internet_service = np.random.choice(['DSL', 'Fiber Optic', 'No'], size=N, p=[0.35, 0.45, 0.2])

# Churn probability - built using a simple weighted logic so clusters actually
# come out meaningful (high charges + low tenure + month-to-month -> high churn risk)
risk_score = (
    (monthly_charges / monthly_charges.max()) * 0.4
    + (1 - tenure_months / tenure_months.max()) * 0.35
    + (support_calls / (support_calls.max() + 1)) * 0.15
    + np.where(contract_type == 'Month-to-Month', 0.1, 0.0)
)

churn_probability = np.clip(risk_score + np.random.normal(0, 0.07, size=N), 0, 1).round(3)

# Final churn label (1 = churned, 0 = stayed) based on probability + some randomness
churn = (churn_probability > 0.55).astype(int)
flip_mask = np.random.rand(N) < 0.05  # 5% noise so the model isn't "too perfect"
churn[flip_mask] = 1 - churn[flip_mask]

df = pd.DataFrame({
    'CustomerID': [f'CUST{i:05d}' for i in range(1, N + 1)],
    'Tenure Months': tenure_months,
    'Monthly Charges': monthly_charges,
    'Total Charges': total_charges,
    'Support Calls': support_calls,
    'Contract Type': contract_type,
    'Internet Service': internet_service,
    'Churn Probability': churn_probability,
    'Churn': churn
})

df.to_csv('customer_churn.csv', index=False)
print(f"customer_churn.csv ban gaya. Shape: {df.shape}")
print(df.head())
