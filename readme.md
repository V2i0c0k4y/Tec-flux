# TecFlux ⚡
### AI-Powered Prediction of Ionospheric Total Electron Content (TEC)

> A hybrid ARIMA-LSTM framework that calculates and forecasts ionospheric TEC using solar and geomagnetic parameters — built to improve GPS, GNSS, and IRNSS satellite navigation accuracy.

---

## What is TecFlux?

The ionosphere distorts satellite signals, causing GPS errors. The key factor is **Total Electron Content (TEC)** — the number of free electrons between a satellite and a receiver.

TecFlux works in two stages:
1. **Calculate TEC** from solar/geomagnetic parameters using the Chapman electron density function
2. **Predict future TEC** using LSTM and ARIMA models trained on the calculated values

---

## How It Works

┌─────────────────────────────────────────┐
│   Solar & Geomagnetic Inputs            │
│   F10.7  │  Sunspot No.  │  Kp  │  Ap  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Chapman Function                │
│  Physics-based TEC calculation          │
│  Integrates electron density            │
│  from 90km → 1000km altitude            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Feature Engineering             │
│  Lag features │ Rolling stats           │
│  Date features │ Interaction terms      │
└────────┬─────────────────┬─────────────┘
         │                 │
         ▼                 ▼
   ┌───────────┐     ┌───────────┐
   │   LSTM    │     │   ARIMA   │
   │  64 units │     │  (5,1,0)  │
   │   Adam    │     │ statsmodel│
   └─────┬─────┘     └─────┬─────┘
         │                 │
         └────────┬────────┘
                  ▼
     ┌─────────────────────┐
     │   TEC (TECU) Output  │
     └──────────┬──────────┘
                ▼
     ┌─────────────────────┐
     │   Flask Web App      │
     │  Real-time Prediction│
     └─────────────────────┘

---

## Dataset

- **Source:** NASA OMNIWeb, NOAA Space Weather Prediction Centre
- **Coverage:** Solar Cycles 19–24 (1954–2018)
- **Input Parameters:**
  - Sunspot Number (SN)
  - F10.7 Solar Flux (observed)
  - Kp Index (Kp1–Kp8, averaged)
  - Ap Index (daily)

---

## Features Used by LSTM

| Feature | Description |
|---------|-------------|
| TEC_Lag1, TEC_Lag2 | Previous TEC values (1 & 2 days back) |
| TEC_Rolling_Mean | 7-day rolling average of TEC |
| TEC_Rolling_Std | 7-day rolling standard deviation |
| Month, Day, Weekday | Date-based temporal features |
| Kp_mean | Average of Kp1–Kp8 indices |
| Sunspot (SN) | Daily sunspot number |
| F107 | F10.7 solar flux |
| Kp_F107 | Interaction term (Kp × F10.7) |

---

## Results

| Metric | Value |
|--------|-------|
| Forecast error | < ±2 TECU |
| Real-time response | < 2 seconds |
| Train/Test split | 80% / 20% |
| Training data | 6 solar cycles (1954–2018) |

---

## Project Structure

```
Tec-flux/
├── Tecflux.ipynb            # Full pipeline: TEC calculation, ARIMA & LSTM training
├── app.py                   # Flask web app for real-time TEC prediction
├── prepared_tec_data.csv    # Processed dataset with engineered features
├── tec_lstm_model.h5        # Trained LSTM model weights
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/V2i0c0k4y/Tec-flux.git
cd Tec-flux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the web app
```bash
python app.py
```
Open `http://localhost:5000` in your browser and enter a date to get the predicted TEC value.

### 4. Explore the notebook
Open `Tecflux.ipynb` in Jupyter to walk through the full pipeline — Chapman function TEC calculation, feature engineering, ARIMA fitting, LSTM training, and evaluation.

---

## Tech Stack

- **Python** — TensorFlow/Keras, Pandas, NumPy, SciPy, Scikit-learn
- **Physics Model** — Chapman function (scipy.integrate)
- **ARIMA** — statsmodels
- **LSTM** — TensorFlow/Keras (64 units, Adam optimizer)
- **Web** — Flask
- **Data Sources** — NASA OMNIWeb, NOAA SWPC

---

## Applications

- GNSS/GPS ionospheric delay correction
- Satellite communication planning
- Space weather monitoring
- Aviation and defence navigation systems

---

## Research Context

Developed as undergraduate research at **SNS College of Technology, Coimbatore**
Department of Artificial Intelligence & Machine Learning

**Reference:**
Han, Y., Wang, L., Fu, W., Zhou, H., Li, T., & Che, R. (2022). Machine Learning-Based Short-Term GPS TEC Forecasting During High Solar Activity and Magnetic Storm Periods. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 15*, 115.

---

## License

Open for academic and research use.
