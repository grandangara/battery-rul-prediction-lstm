import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# --- PART 1: VERSION-COMPATIBLE DATA LOADER ---
def load_random_walk_data(mat_file_path):
    print(f"Loading {mat_file_path} (Final Mode)...")
    try:
        mat = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"ERROR: '{mat_file_path}' not found.")
        return None

    # Find the main data key
    key_name = [key for key in mat.keys() if not key.startswith('_')][0]
    steps = mat[key_name][0, 0]['step'][0]

    # Get column names dynamically
    sample = steps[0]
    names = sample.dtype.names

    def get_col(name_part):
        for n in names:
            if name_part.lower() in n.lower(): return n
        return None

    k_vol = get_col('voltage')
    k_cur = get_col('current')
    k_time = get_col('relative') # relativeTime
    k_temp = get_col('temp')     # temperature

    extracted_data = []
    print(f"Scanning {len(steps)} total steps...")

    for i in range(len(steps)):
        step_data = steps[i]

        try:
            # Extract and flatten data
            raw_v = step_data[k_vol][0].flatten()
            raw_c = step_data[k_cur][0].flatten()
            raw_t = step_data[k_time][0].flatten()

            # Duration Calculation
            duration = raw_t[-1] - raw_t[0]
            avg_current = np.mean(raw_c)

            # FILTER: Long duration (>2000s) and Positive Current (>0.5A)
            # This identifies "Reference Discharge" cycles in RW datasets
            if duration > 2000 and avg_current > 0.5:

                # --- COMPATIBILITY FIX ---
                # NumPy 2.0 compatibility: 'trapz' was removed, 'trapezoid' was added.
                # We try both to ensure the code runs on any numpy version.
                try:
                    # For older NumPy versions
                    integration_result = np.trapz(np.abs(raw_c), raw_t)
                except AttributeError:
                    # For new NumPy 2.0+ versions
                    integration_result = np.trapezoid(np.abs(raw_c), raw_t)

                capacity_ah = integration_result / 3600.0

                # Temperature Handling
                if k_temp:
                    raw_temp = step_data[k_temp][0].flatten()
                    avg_temp = np.mean(raw_temp)
                else:
                    avg_temp = 24.0 # Default room temp if missing

                extracted_data.append({
                    'Avg_Voltage': np.mean(raw_v),
                    'Avg_Current': avg_current,
                    'Avg_Temp': avg_temp,
                    'Capacity': capacity_ah
                })
                print(f"-> Step {i} ADDED! (Capacity: {capacity_ah:.2f} Ah)")

        except Exception as e:
            # If error occurs, show only the first one to avoid spamming
            if i == 0: print(f"Error detail: {e}")
            continue

    df = pd.DataFrame(extracted_data)

    if df.empty:
        print("\nCRITICAL: Data list remained empty. Check file or filters.")
        return None

    print(f"\nSUCCESS: {len(df)} reference data points collected.")
    return df


# --- PART 2: AI MODEL (LSTM) ---

df = load_random_walk_data('RW9.mat')

if df is None:
    print("Program stopping because data could not be loaded.")
    exit()

# Prepare Data
raw_data = df[['Avg_Voltage', 'Avg_Current', 'Avg_Temp', 'Capacity']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(raw_data)

X_all = scaled_data[:, 0:3] # Input: Voltage, Current, Temp
y_all = scaled_data[:, 3]   # Output: Capacity


# Sliding Window Function
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


TIME_STEPS = 2
X_lstm, y_lstm = create_dataset(X_all, y_all, TIME_STEPS)

# Train/Test Split
train_size = int(len(X_lstm) * 0.70)
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

# Build LSTM Model
model = Sequential()
model.add(LSTM(32, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
print("Training AI Model...")
model.fit(X_train, y_train, epochs=80, batch_size=2, verbose=0)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# Inverse Transform for Visualization
def inverse_transform_prediction(prediction):
    dummy = np.zeros((len(prediction), 4))
    dummy[:, 3] = prediction[:, 0]
    return scaler.inverse_transform(dummy)[:, 3]


y_train_inv = inverse_transform_prediction(train_predict)
y_test_inv = inverse_transform_prediction(test_predict)
y_actual_inv = scaler.inverse_transform(scaled_data)[:, 3]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_actual_inv, label='Ground Truth (RW9)', color='purple', marker='o', alpha=0.5)
plt.plot(np.arange(TIME_STEPS, len(y_train_inv) + TIME_STEPS), y_train_inv, label='Training Phase', color='orange')
plt.plot(np.arange(len(y_train_inv) + TIME_STEPS, len(y_train_inv) + TIME_STEPS + len(y_test_inv)), y_test_inv,
         label='AI Prediction', color='green', linewidth=2)
plt.title('Random Walk Battery Life Prediction')
plt.xlabel('Reference Test Count')
plt.ylabel('Capacity (Ah)')
plt.legend()
plt.grid(True)
plt.show()