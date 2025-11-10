# Week-2
This project develops a Generative AI-based predictive maintenance system for Electric Vehicles. Using sensor and operational data, the model (VAE + LSTM) simulates future failure patterns of components like batteries, motors, and controllers, enabling early fault detection, improved reliability, and optimized maintenance scheduling.

# 🚗 Generative AI for Predictive Maintenance in Electric Vehicles

## 🧠 Problem Statement
Develop a **Generative AI model** that can **predict and simulate possible future failure modes** of Electric Vehicle (EV) components such as the **battery, motor, and controller** by analyzing **sensor data** and **operational history**.  
The goal is to **enable proactive maintenance scheduling**, **reduce unexpected breakdowns**, and **improve the overall reliability and safety** of EV systems.

---

## 🎯 Objectives
- Build a predictive model to forecast EV component failures using sensor and historical operational data.
- Simulate potential failure scenarios using generative models (e.g., Variational Autoencoders, GANs).
- Enable proactive maintenance planning through early warning systems.
- Minimize downtime and maintenance costs while improving system reliability.

---

## ⚙️ Methodology
1. **Data Collection & Preprocessing**
   - Collect EV telemetry data (battery temp, current, voltage, vibration, speed, etc.).
   - Clean, normalize, and label data for component health and failure events.

2. **Feature Engineering**
   - Extract temporal, statistical, and frequency-domain features.
   - Compute rolling averages, deltas, and trend features.

3. **Model Development**
   - **Predictive Model:** Random Forest, XGBoost, or LSTM for predicting failures / RUL.
   - **Generative Model:** Autoencoder or GAN for simulating potential failure patterns.

4. **Evaluation Metrics**
   - **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC.
   - **Regression:** MAE, MSE, RMSE, R².
   - **Goal:** Achieve >80% accuracy and high recall for failure prediction.

5. **Deployment**
   - Deploy trained model into a maintenance dashboard.
   - Visualize health scores and simulated failure timelines.

---

## 📊 Dataset Overview
- **Source:** EV telemetry dataset (`EV_d.csv`)
- **Example features:**
  - Battery Temperature (°C)
  - Motor Current (A)
  - Voltage (V)
  - Speed (km/h)
  - Vibration Sensor Data
  - Controller Temperature (°C)
  - Failure Label / Remaining Useful Life (RUL)

---

## 🧪 Model Evaluation
| Metric | Description | Target |
|---------|--------------|--------|
| Accuracy | Overall prediction correctness | ≥ 80% |
| Recall | Ability to catch failure events | ≥ 90% |
| MSE / MAE | Prediction error for RUL | As low as possible |
| R² | Variance explained by regression model | ≥ 0.8 |

---

## 🧰 Tools & Technologies
- **Languages:** Python  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn  
- **ML Models:** RandomForest, XGBoost, LSTM, Autoencoder, GAN  
- **Evaluation:** Accuracy, Recall, MSE, MAE  
- **Environment:** Jupyter Notebook / VS Code / Google Colab  

---
