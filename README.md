# battery-rul-prediction-lstm
🔋 Physics-Informed LSTM for Battery RUL Prediction (NASA Dataset)
📌 Project Overview
This project implements a Deep Learning (LSTM) model to predict the Remaining Useful Life (RUL) and capacity degradation of Lithium-Ion batteries.

Unlike standard datasets with constant load profiles, this project utilizes NASA's Randomized Battery Usage Data Set (RW9), which simulates real-world driving conditions (Random Walk). The model successfully filters chaotic data using physics-based rules and predicts future battery capacity with high accuracy.

🎯 Key Objectives

Real-World Simulation: Handling stochastic (random) load profiles instead of fixed lab conditions.

Physics-Informed Data Engineering: Developed a robust data loader that filters "Reference Discharge" cycles based on physical properties (Duration > 15 mins, Positive Current) rather than unreliable file labels.

Version Compatibility: Solved critical numpy version conflicts (trapz vs trapezoid) via dynamic exception handling.

📊 Results
The model was trained on the first 70% of the reference cycles and tested on the remaining 30%.

Purple Points: Real ground-truth capacity (derived from NASA RW9 via Coulomb Counting).

Green Line: The model's prediction for unseen future cycles.

(Please upload your 'result_graph.png' to the repository to see the image here)

🛠️ Technologies & Methodology
Tech Stack

Python: Core logic.

TensorFlow / Keras: LSTM (Long Short-Term Memory) neural network construction.

SciPy: Parsing complex MATLAB (.mat) data structures.

Pandas & NumPy: Data manipulation and numerical integration (Coulomb Counting).

Matplotlib: Visualization of degradation trends.

The Pipeline

Smart Data Loading: A custom script reads raw NASA .mat files. It ignores inconsistent string labels and uses Physics-based Logic:

Condition: Duration > 900s AND Avg Current > 0.5A = Reference Discharge Cycle.

Feature Engineering: Extracts Average Voltage, Current, Temperature, and calculates Capacity (Ah) via integration.

Normalization: Scales all features to [0, 1] range using MinMaxScaler.

Sliding Window: Prepares time-series data (looking back N steps to predict N+1).

LSTM Modeling: A sequential model with LSTM layers learns the non-linear degradation path.
🚀 How to Run
Clone the Repository

Bash
git clone https://github.com/your-username/battery-rul-prediction.git
cd battery-rul-prediction
Install Dependencies

Bash
pip install -r requirements.txt
Download Dataset

Download RW9.mat from the NASA Prognostics Data Repository (Dataset #11).

Place RW9.mat in the project root folder.

Run the Prediction

Bash
python nasa_final_project.py
🧠 Why This Matters?
Accurately predicting battery health in random usage scenarios is critical for:

EV Range Estimation: Reducing "range anxiety" for drivers.

Second-Life Assessment: Determining if an old EV battery can be reused in solar energy storage.

Predictive Maintenance: Preventing failures before they happen in critical energy systems.
