from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the dataset and preprocess
df = pd.read_csv('prepared_tec_data.csv')
target = 'TEC (TECU)'
features = ['TEC_Lag1', 'TEC_Lag2', 'TEC_Rolling_Mean', 'TEC_Rolling_Std', 'Month', 'Day', 'Weekday', 'Kp_mean', 'Sunspot', 'F107', 'Kp_F107']

X = df[features].values
y = df[target].values

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_Y.fit_transform(y.reshape(-1, 1))

# LSTM input reshaping
train_size = int(len(df) * 0.8)
X_train = X_scaled[:train_size].reshape(-1, 1, X.shape[1])
y_train = y_scaled[:train_size]
X_test = X_scaled[train_size:].reshape(-1, 1, X.shape[1])
y_test = y_scaled[train_size:]

# Load or train the model
try:
    lstm_model = load_model('tec_lstm_model(3).h5')
except:
    def build_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(units=32, activation='relu', input_shape=input_shape))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    lstm_model = build_lstm_model((1, X.shape[1]))
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
    lstm_model.save('tec_model.h5')

# Prediction function
def predict_tec(user_date):
    user_date = pd.to_datetime(user_date)
    user_input_dict = {
        'TEC_Lag1': 10,
        'TEC_Lag2': 12,
        'TEC_Rolling_Mean': 11,
        'TEC_Rolling_Std': 2,
        'Month': user_date.month,
        'Day': user_date.day,
        'Weekday': user_date.weekday(),
        'Kp_mean': 3,
        'Sunspot': 50,
        'F107': 150,
        'Kp_F107': 4
    }
    input_array = np.array(list(user_input_dict.values())).reshape(1, -1)
    input_scaled = scaler_X.transform(input_array)
    input_reshaped = input_scaled.reshape(1, 1, -1)
    pred_scaled = lstm_model.predict(input_reshaped)
    pred_actual = scaler_Y.inverse_transform(pred_scaled)
    return pred_actual[0][0]

# Plot generator
def generate_plot(user_date, predicted_tec):
    past_predicted_tecs = [125.1, 100.8, 110.9, 130.8, 134.9]
    dates = pd.date_range(end=pd.to_datetime(user_date), periods=6).strftime('%Y-%m-%d').tolist()
    all_preds = past_predicted_tecs + [predicted_tec]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, all_preds, marker='o', color='green', label='Predicted TEC')
    plt.axvline(x=dates[-1], color='red', linestyle='--', label='User Date')
    plt.title('Predicted TEC Including User Input Date')
    plt.xlabel('Date')
    plt.ylabel('TEC (TECU)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    return img_data

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_tec = None
    plot_url = None

    if request.method == 'POST':
        date_input = request.form['date']
        predicted_tec = round(predict_tec(date_input), 2)
        plot_url = generate_plot(date_input, predicted_tec)

    return render_template('index.html', predicted_tec=predicted_tec, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
