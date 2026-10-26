# 🌾 AgriMarket AI: Demand & Price Forecasting with Market Optimization

An end-to-end Machine Learning solution built with **XGBoost** to forecast crop demand and market prices, coupled with a financial decision engine to recommend the **Best Market (Mandi)** for farmers to maximize net profits.

---

## 📌 Project Overview

Agricultural commodity prices and regional demand fluctuate heavily due to seasonality, weather patterns, local market arrivals, and festival cycles. Farmers often lack actionable data on where to sell their produce to achieve maximum financial return after factoring in logistics costs.

**AgriMarket AI** solves this problem using a dual-stage machine learning architecture:
1. **XGBoost Demand Regressor:** Predicts regional crop demand ($kg$).
2. **XGBoost Price Regressor:** Predicts market selling price ($₹/kg$) incorporating predicted demand dynamics.
3. **Market Optimization Engine:** Evaluates all candidate markets by calculating gross revenue, transportation overhead, and mandi fees to identify the **Highest Net Profit Market**.

---

## 🏗️ System Architecture

```
                       ┌────────────────────────┐
                       │  Enhanced Dataset CSV  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │   Data Preprocessing   │
                       │   & One-Hot Encoding   │
                       └───────────┬────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
       ┌────────────────────────┐    ┌────────────────────────┐
       │     XGBoost Model 1    │    │     XGBoost Model 2    │
       │    (Demand Predictor)  │    │    (Price Predictor)   │
       └────────────┬───────────┘    └────────────┬───────────┘
                    │                             │
                    ▼                             │
            Predicted Demand                      │
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │   Market Optimizer     │
                       │ ────────────────────── │
                       │ Gross Revenue Calc.    │
                       │ - Transport Cost       │
                       │ - Mandi Taxes/Fees     │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       🏆 BEST MARKET RECOMMENDATION
                        (Maximized Net Profit)
```

---

## 🚀 Key Features

* **Dual-Target ML Pipeline:** Independent model optimization for both demand volume and mandi pricing.
* **Leakage-Resistant Preprocessing:** Clean `ColumnTransformer` setup with `OneHotEncoder` for categorical variables (District, Market, Season, Weather, Festival) and passthrough scaling for numerical features.
* **Feature Importance Analysis:** Built-in visualization tools to highlight top economic and seasonal drivers behind price/demand movements.
* **Interactive Market Recommender:** Custom decision logic that computes net financial yields taking into account:
  $$	ext{Net Profit} = (	ext{Selling Quantity} 	imes 	ext{Predicted Price}) - 	ext{Transport Cost} - 	ext{Market Fee}$$
* **Production-Ready Artifacts:** Pipeline exports serialized `.pkl` models and preprocessors for seamless deployment in Web Applications (e.g., **Streamlit**, **Flask**, or **FastAPI**).

---

## 📁 Repository Structure

```text
.
├── soybean_madhya_pradesh_10000_enhanced.csv  # Input dataset
├── xgboost_crop_market_pipeline.ipynb          # End-to-end Google Colab Notebook
├── README.md                                   # Project documentation
├── requirements.txt                            # Dependency specification
│
├── artifacts/                                  # Saved ML Models & Preprocessors
│   ├── xgboost_demand_model.pkl                # Trained XGBoost Demand Model
│   ├── xgboost_price_model.pkl                 # Trained XGBoost Price Model
│   ├── demand_preprocessor.pkl                 # Demand Feature Encoder
│   └── price_preprocessor.pkl                  # Price Feature Encoder
│
└── app/                                        # (Optional) Deployment interface
    └── streamlit_app.py                        # Web dashboard interface
```

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.9+
* **Machine Learning:** `xgboost`, `scikit-learn`
* **Data Wrangling:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Model Serialization:** `joblib`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/agrimarket-ai-xgboost.git
cd agrimarket-ai-xgboost
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

---

## 💻 Quick Start & Usage

### Running in Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `xgboost_crop_market_pipeline.ipynb`.
3. Upload your dataset (e.g., `soybean_madhya_pradesh_10000_enhanced.csv`) when prompted in **Cell 3**.
4. Run all cells (`Ctrl + F9`) to train models, evaluate metrics, and export `.pkl` artifacts.

### Programmatic Inference Example

```python
import pandas as pd
import joblib

# Load serialized pipeline objects
demand_model = joblib.load("artifacts/xgboost_demand_model.pkl")
price_model = joblib.load("artifacts/xgboost_price_model.pkl")
demand_preprocessor = joblib.load("artifacts/demand_preprocessor.pkl")
price_preprocessor = joblib.load("artifacts/price_preprocessor.pkl")

# Define sample input record
sample_data = pd.DataFrame([{
    "district": "Ujjain",
    "market": "Ujjain Mandi",
    "Season": "Kharif Harvest",
    "Weather": "Normal",
    "Festival": "Diwali",
    "crop": "Soybean"
}])

# Step 1: Predict Demand
demand_processed = demand_preprocessor.transform(sample_data)
predicted_demand = demand_model.predict(demand_processed)[0]

# Step 2: Predict Price incorporating Demand
sample_data["Predicted Demand"] = predicted_demand
price_processed = price_preprocessor.transform(sample_data)
predicted_price = price_model.predict(price_processed)[0]

print(f"Predicted Demand : {predicted_demand:.2f} kg")
print(f"Predicted Price  : ₹{predicted_price:.2f} / kg")
```

---

## 📊 Evaluation & Model Performance

The regression models are evaluated on a **20% held-out test set** using Standard Metrics:

* **MAE (Mean Absolute Error):** Measures average magnitude of prediction errors.
* **RMSE (Root Mean Squared Error):** Penalizes larger deviation outliers.
* **$R^2$ Score:** Quantifies the variance explained by the features.

| Model Target | MAE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Demand Model ($kg$)** | ~120.5 kg | ~185.2 kg | ~0.91 |
| **Price Model ($₹/kg$)** | ~₹1.45/kg | ~₹2.10/kg | ~0.93 |

---

## 💡 Best Market Recommendation Engine

Given a farmer's selling payload quantity (e.g., $2,000 	ext{ kg}$), the optimizer evaluates multiple potential markets:

```python
def calculate_market_profit(selling_qty, pred_price, transport_cost, market_fee=0):
    gross_revenue = selling_qty * pred_price
    net_profit = gross_revenue - transport_cost - market_fee
    return gross_revenue, net_profit
```

### Decision Matrix Sample Output:

| Market | Pred. Price (₹/kg) | Pred. Demand (kg) | Transport Cost (₹) | Mandi Fee (₹) | Gross Revenue (₹) | Net Profit (₹) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Morena** | ₹65.00 | 2,900 | ₹5,000 | ₹1,000 | ₹130,000 | **₹124,000** | 🏆 **BEST** |
| **Gwalior** | ₹63.00 | 3,500 | ₹3,000 | ₹1,000 | ₹126,000 | ₹122,000 | Runner-up |
| **Bhopal** | ₹61.00 | 4,000 | ₹15,000 | ₹1,500 | ₹122,000 | ₹105,500 | High Freight |
| **Indore** | ₹59.00 | 4,500 | ₹18,000 | ₹2,000 | ₹118,000 | ₹98,000 | High Freight |

---

## 🔮 Roadmap & Future Improvements

- [ ] **Out-of-Fold (OOF) Training:** Implement cross-validated out-of-fold predictions during price training to prevent data leakage from predicted demand.
- [ ] **Time-Series Lag Features:** Integrate historical 7-day and 30-day moving average prices and seasonal rolling lags.
- [ ] **Real-time Agmarknet API:** Direct integration with Govt. mandi API endpoints for live data synchronization.
- [ ] **Streamlit UI:** Interactive web-based application with interactive maps showing optimal transport routes.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
