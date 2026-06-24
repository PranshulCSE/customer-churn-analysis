# Customer Churn Analysis Project

Telecom customer data ko analyze karke do kaam kiye gaye hain:

1. **Customer Segmentation** - K-Means clustering se customers ko 3 groups
   mein baata gaya hai (Loyal Low-Risk, Moderate Risk, High Risk) based on
   unke charges, tenure aur churn probability ke pattern.
2. **Churn Prediction** - Random Forest classifier se predict kiya gaya hai
   ki kaunsa customer company chhod sakta hai, with hyperparameter tuning
   to find the best model.

## Folder Structure

```
customer_churn_project/
├── data/
│   ├── generate_data.py      -> synthetic dataset banata hai
│   └── customer_churn.csv    -> generate hone ke baad yahan ban jayega
├── churn_analysis.py         -> main script (clustering + classification)
├── outputs/                  -> sare plots aur results yahan save hote hain
├── requirements.txt
└── README.md
```

## How to Run

1. Virtual environment banao (optional, recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

2. Dependencies install karo:
   ```
   pip install -r requirements.txt
   ```

3. Dataset generate karo (sirf pehli baar):
   ```
   cd data
   python generate_data.py
   cd ..
   ```
   Isse `data/customer_churn.csv` ban jayega.

4. Main analysis chalao:
   ```
   python churn_analysis.py
   ```

5. Sare outputs (plots, hyperparameter results CSV) `outputs/` folder mein
   save ho jayenge:
   - `elbow_method.png`
   - `segment_monthly_charges.png`
   - `segment_tenure.png`
   - `segment_total_charges.png`
   - `feature_importance.png`
   - `hyperparameter_results.csv`

## Notes

- Dataset synthetic hai (real customer data ke jagah generate kiya gaya hai)
  taaki project standalone run ho sake bina kisi external download ke.
  Same structure ke saath ise asaani se real telecom dataset (jaise Kaggle's
  Telco Customer Churn) pe bhi use kiya ja sakta hai - bas column names
  match karne honge.
- Cluster names automatic decide hote hain based on average churn
  probability per cluster, isliye dataset change hone par bhi naming
  consistent rahegi.
- Recall ko priority diya gaya hai hyperparameter selection mein kyunki
  business ke perspective se churn customer ko miss karna zyada costly
  hai compared to false alarm dena.
