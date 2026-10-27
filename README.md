# F1 Pit-Wall Telemetry & Strategy Suite 🏎️

An enterprise-grade Formula 1 race engineering, telemetry analysis, and strategy workstation powered by **FastF1, Streamlit, SciPy, Scikit-Learn, and Plotly**.

Features authentic GPS cubic-spline track geometries, distance-aligned teammate spatial deltas, empirical tyre thermal-degradation modeling, a Random Forest pit-stop event classifier, and 10,000-run Monte Carlo stochastic race-strategy engines.

---

## 🌟 Key Features

1. **Authentic 2D GPS Circuit Replay (`circuits.py`)**:
   - High-density Cubic Spline trajectory interpolation across iconic Grand Prix circuits (Monza, Silverstone, Spa-Francorchamps, COTA, Monaco, Suzuka, Bahrain, Interlagos).
   - Landmark corner apex annotations and 50Hz rolling CAN Bus telemetry stream.

2. **High-Precision Spatial Telemetry Overlay**:
   - Distance-interpolated speed, throttle, brake, and gear traces between teammates using `scipy.interpolate.interp1d`.
   - Numerical time delta integration and apex corner speed extraction.

3. **Tyre Degradation & Thermal Cliff Modeling**:
   - Combines fuel-corrected compound wear factors (Soft C4, Medium C3, Hard C2) with non-linear exponential thermal decay.
   - Calculates real-time tyre life remaining percentage and strategic cliff warning thresholds.

4. **Machine Learning Pit Stop Classifier**:
   - Stint-level feature engineering (`LapNumber`, `TyreLife`, `PaceDelta`, `CompoundCode`).
   - `RandomForestClassifier(class_weight='balanced')` predicting lap-by-lap pit probability $P(\text{Pit on Lap } L)$ and undercut feasibility.

5. **10,000-Iteration Monte Carlo Strategy Engine**:
   - Simulates full-race outcomes comparing Strategy A (1-Stop: Medium ➔ Hard) vs. Strategy B (2-Stop: Soft ➔ Medium ➔ Medium).
   - Accounts for stochastic safety car / VSC probabilities, pit lane transit deltas, and finishing position distributions.

6. **Dual Interfaces**:
   - **Streamlit Web Application (`app.py`)**: Interactive dashboards, Plotly charts, live tabs, and strategy matrix.
   - **Standalone Desktop HUD (`pitwall_desktop.py`)**: Real-time Matplotlib animation engine with interactive track switching.

---

## 🚀 Quick Start & Installation

### 1. Clone or Download Repository
```bash
git clone https://github.com/<your-username>/f1_pitwall_suite.git
cd f1_pitwall_suite
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```

### 4. Launch Desktop Animation HUD (Optional)
```bash
python pitwall_desktop.py
```

---

## 📂 Project Structure

```
f1_pitwall_suite/
├── app.py                # Main Streamlit web application & dashboards
├── circuits.py           # Circuit geometry engine & cubic spline database
├── pitwall_desktop.py    # Standalone desktop Matplotlib animation HUD
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git exclusion rules (cache & virtual environments)
```

---

## 🛠️ Tech Stack

- **Core & Data:** Python 3.15, Pandas, NumPy
- **Telemetry & F1 APIs:** FastF1, Ergast / Jolpica API Mirror
- **Mathematics & Machine Learning:** SciPy (CubicSpline, interp1d), Scikit-Learn (Random Forest)
- **UI & Visualization:** Streamlit, Plotly Express & Graph Objects, Matplotlib

---

## 👤 Author

**Navneet Prasad**  
- 3rd Year B.Tech Computer Science & Engineering (Data Science) Student  
- GGSESTC Bokaro (Class of 2028)  

