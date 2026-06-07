# Traffic Demand Forecasting

**Team:** Aaryan Dwivedi · Devansh Bhardwaj · Rahul Das · Aagman Khantwal

**Score = 99.9581 / 100**

---

## Project Layout

```
traffic_demand_project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── solution.py
├── requirements.txt
├── README.md
└── submission.csv          ← generated after running solution.py
```

---

## Setup & Run

**Step 1 — Install Python 3.8+**
Download from https://www.python.org/downloads/ and make sure to tick **"Add Python to PATH"**.

**Step 2 — Open terminal / command prompt**
- Windows: `Win + R` → type `cmd` → Enter
- Mac/Linux: open Terminal

**Step 3 — Navigate to the project folder**
```bash
cd path/to/traffic_demand_project
```

**Step 4 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 5 — Run**
```bash
python solution.py
```

The script takes **3–5 minutes**. When finished, `submission.csv` is ready to upload.

---

## Expected Output

```
Train: (77299, 11)  |  Test: (41778, 10)
X_train: (62312, 42)  |  X_test: (41778, 42)
NaN check — train: 0  test: 0

───────────────────────────────────────────────────────
  Fold 1/5  |  R² = 0.999580  |  trees = 5750
  Fold 2/5  |  R² = 0.999619  |  trees = 2767
  Fold 3/5  |  R² = 0.999574  |  trees = 2726
  Fold 4/5  |  R² = 0.999567  |  trees = 2664
  Fold 5/5  |  R² = 0.999550  |  trees = 3216
───────────────────────────────────────────────────────
  OOF R²  = 0.999578
  Score   = 99.9578 / 100
───────────────────────────────────────────────────────

Saved → submission.csv
```

---

## Approach

The dataset gives us:
- `train.csv` — full day 48 (all 96 time-slots) + day 49 early morning (00:00–02:00)
- `test.csv` — day 49 (02:15–13:45), demand unknown

For each location (`geohash`) and time-slot, the strongest predictor of day-49 demand is the day-48 demand at the same slot, scaled by a per-location factor derived from how the morning of day 49 compares to the morning of day 48:

```
scale[geohash] = mean(day49_morning_demand) / mean(day48_morning_demand)
prediction     = day48_slot_demand × scale[geohash]
```

On top of this lookup, LightGBM is trained with 42 features including:
- Per-geohash × slot demand from day 48
- Normalised time-of-day demand patterns
- Day-49 morning statistics (mean, std, slope across slots 0–8)
- Cyclic hour/slot encodings, road type, weather, lanes, temperature

The model is validated by simulating the exact test scenario on day 48 (use slots 0–8 as morning context, predict slots 9–95 where ground truth is available).

---

## Dependencies

| Package | Version |
|---|---|
| pandas | ≥ 1.5 |
| numpy | ≥ 1.23 |
| scipy | ≥ 1.9 |
| scikit-learn | ≥ 1.1 |
| lightgbm | ≥ 3.3 |
