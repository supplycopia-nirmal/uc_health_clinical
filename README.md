# UC Health Clinical Intelligence & Variation Platform

A multi-tenant clinical variation and supply chain intelligence platform developed for **UC Health**, built on modern Snowflake cloud consumption analytics, machine learning anomaly detection, frequent pattern mining (FP-Growth), and OpenAI clinical evaluation.

---

## Architecture Overview

This platform is completely independent, decoupled from legacy implementations, and parameterized for multi-tenancy.

```
uc_health_clinical/
├── config.py                         # Multi-tenant settings & Snowflake RSA key manager
├── pipeline/
│   ├── data_pipeline.py              # Extraction, Isolation Forest, Compositional Outlier & BOM Generator
│   └── generate_master_constructs.py # FP-Growth Construct Miner with True Case Support & RLHF
├── server/
│   └── main.py                       # FastAPI analytical backend with ORJSON & GZip
├── web/
│   ├── index.html                    # Responsive UC Health branded SPA UI
│   └── static/
│       └── uc_health_logo.svg        # UC Health vector branding
├── data/                             # Local artifact storage (Parquet/CSV)
├── Dockerfile                        # Production container specification
└── requirements.txt                  # Python dependencies
```

---

## Features & Analytical Capabilities

1. **Executive Dashboard & Variance Analysis**:
   - Volume vs Cost per surgeon scatter plots.
   - Outlier detection bubble charts (Cost vs Item volume).
   - Unified variance breakdown (Standard, Non-Standard, Clinical Variance, Omitted).
   - Top 10 & Bottom 10 clinical spend categories.
   - 5-level hierarchical spend sunburst (Spend Category → Contract Category → Subcategory → Brand → Description).
2. **AI Clinical & Supply Chain Insights**:
   - Executive-level clinical variation summaries generated using OpenAI (`gpt-5.6-luna` / `gpt-4o`).
3. **Clinical Substitutions & Equivalence**:
   - Discovers candidate brand equivalents within UNSPSC subcategories and models annual savings opportunities.
4. **Pack & Kit Opportunities (Association Rule Mining)**:
   - Identifies co-utilized items using FP-Growth frequent itemset mining.
5. **Physician Preference Cards**:
   - Automatically builds Open Lists (>70% historical utilization) and Hold Lists (30-70% utilization).
   - Flags brand conflicts against facility standards.
   - Integrated Reinforcement Learning from Human Feedback (RLHF) loop evaluated by OpenAI.
6. **Compare Preference Cards**:
   - Side-by-side comparison across surgeons and against the hospital master standard.
7. **Facility Price Parity**:
   - Analyzes unit price disparities across 12 UC Health facilities (UH OR, West Chester, Drake Center, etc.) and highlights immediate contract renegotiation savings.
8. **Contract Compliance & Savings Leakage**:
   - Quantifies off-contract spend and potential savings opportunities using pre-calculated engine variance.
9. **Predictive Case Cost Modeling (XGBoost)**:
   - Machine learning regression model predicting case cost based on patient age, ASA physical status rating, BMI bucket, smoking status, diabetic status, and patient type.
10. **The Bee (AI Clinical Copilot)**:
    - Conversational voice & text clinical intelligence assistant.

---

## Cloud Tables Provisioned in Snowflake (`SUPPLYCOPIA_TRANSFORM.SCRETAIL_IMPORT`)

* `UC_HEALTH_HYBRID_ANOMALY_DETECTION`: 1,378,322 rows with Isolation Forest and Compositional Outlier flags.
* `UC_HEALTH_STANDARD_BOMS_REPORT`: 3,092 standard Bill of Materials items across procedures.
* `UC_HEALTH_MASTER_SUBSTITUTION_MATRIX`: Mined maximal constructs across all historical cases.
* `UC_HEALTH_PHYSICIAN_SUBSTITUTION_MATRIX`: Surgeon-specific clinical constructs.

---

## Quickstart & Execution

### 1. Prerequisites
- Python 3.10+
- Snowflake RSA Private Key (`lambda_migration._rsa_key.p8`)

### 2. Run the Data Pipeline
```bash
python pipeline/data_pipeline.py
```

### 3. Run the Construct Mining Engine
```bash
python pipeline/generate_master_constructs.py
```

### 4. Launch the Application Server
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000` and log in:
- **Username**: `uchealth_admin`
- **Password**: `clinical_copia`
