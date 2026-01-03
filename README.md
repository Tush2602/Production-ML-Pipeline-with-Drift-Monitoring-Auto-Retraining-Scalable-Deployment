# 🚀 Production ML Pipeline with Drift Monitoring, Auto-Retraining & Scalable Deployment

This project implements a **production-grade Machine Learning system** for customer churn prediction, covering the **full ML lifecycle** — from data ingestion to scalable deployment with monitoring and automatic retraining.

Unlike notebook-only projects, this system is **modular, deployable, monitorable, and retrainable**, closely mirroring how real ML systems operate in industry.

---

## 📌 Key Features

* **End-to-End ML Pipeline**

  * Data ingestion, preprocessing, training, evaluation
  * Artifact persistence (model, preprocessor, encoders, metrics)

* **Inference Pipeline**

  * Stateless prediction service
  * Threshold-based decision logic
  * Clean separation from training logic

* **FastAPI Deployment**

  * Typed request/response schemas
  * Swagger UI for testing
  * API-first design

* **Drift Monitoring**

  * Statistical data drift detection using **Kolmogorov–Smirnov (KS) test**
  * Feature-wise drift analysis

* **Auto-Retraining**

  * Automatic model retraining triggered when drift exceeds threshold
  * Centralized retraining via training pipeline

* **Scalable Deployment**

  * Dockerized inference service
  * Horizontally scalable by running multiple containers

---

## 🏗️ System Architecture (High-Level)

```
           ┌──────────────┐
           │   Raw Data   │
           └──────┬───────┘
                  ↓
        ┌────────────────────┐
        │ Training Pipeline  │
        │ (Ingestion → ML)   │
        └──────┬─────────────┘
               ↓
        Saved Model Artifacts
               ↓
        ┌────────────────────┐
        │ Inference Pipeline │
        └──────┬─────────────┘
               ↓
            FastAPI
               ↓
        ┌────────────────────┐
        │ Drift Detection    │
        └──────┬─────────────┘
               ↓
      Auto-Retraining Trigger
```

---

## 📂 Project Structure

```
src/
├── api/                    # FastAPI layer
│   ├── app.py
│   └── schema.py
│
├── pipeline/               # Orchestration
│   ├── training_pipeline.py
│   └── inference_pipeline.py
│
├── monitoring/             # Monitoring & retraining
│   ├── drift_detection.py
│   └── retraining_trigger.py
│
├── utils/                  # Logging & exceptions
├── config/                 # Path configurations
├── data_ingestion.py
├── preprocessing.py
├── train.py
├── evaluate.py
└── predict.py
│
docker/
└── Dockerfile.api
│
data/
├── raw/
├── processed/
├── artifacts/
└── drift/
```

---

## ⚙️ Tech Stack

* **Language:** Python 3.10
* **ML:** Scikit-learn, NumPy, Pandas
* **API:** FastAPI, Pydantic, Uvicorn
* **Monitoring:** SciPy (KS test)
* **Deployment:** Docker
* **Logging & Errors:** Custom logger + exception handling

---

## 🚀 How to Run

### 1️⃣ Run Locally (Without Docker)

```bash
pip install -r requirements.txt
uvicorn src.api.app:app --reload
```

Open:

```
http://localhost:8000/docs
```

---

### 2️⃣ Run with Docker (Scalable Deployment)

#### Build image

```bash
docker build --no-cache -f docker/Dockerfile.api -t churn-api .
```

#### Run container

```bash
docker run -p 8000:8000 churn-api
```

Open:

```
http://localhost:8000/docs
```

---

## 📊 Drift Detection & Auto-Retraining

* **Data Drift Detection**

  * Uses KS test on numerical features
  * Compares training baseline vs recent inference data
  * Drift detected when a significant portion of features change distribution

* **Auto-Retraining**

  * Triggered automatically when drift threshold is crossed
  * Retraining executed via centralized `training_pipeline.py`
  * No manual intervention required

To simulate drift:

1. Add new data to:

   ```
   data/drift/current_data.csv
   ```
2. Run:

   ```bash
   python -m src.monitoring.retraining_trigger
   ```

---

## 🎯 Why This Project Matters

This project demonstrates:

* Real **ML system design**, not just modeling
* Production-ready inference separation
* Monitoring and reliability awareness
* Scalable deployment mindset

It reflects **how ML systems are built and maintained in industry**, not just how models are trained.

---

## 🔮 Future Extensions

* Kubernetes orchestration
* MLflow experiment tracking
* Streaming inference (Kafka)
* Feature stores
* Cloud deployment (AWS/GCP)


---

## ✅ Project Status

**✔ COMPLETE — Production-Ready ML System**

