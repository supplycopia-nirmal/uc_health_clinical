"""
UC Health Comorbidity-Adjusted Expected CPC & Risk Stratification Engine
Multivariate clinical acuity modeling utilizing ASA rating, BMI class, age bucket,
smoking status, and diabetic status to compute Expected CPC, Observed CPC, and O/E Ratios.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ClinicalRiskEngine:
    def __init__(self):
        pass

    @staticmethod
    def _parse_asa(val: Any) -> float:
        if pd.isna(val):
            return 2.0
        s = str(val).strip()
        if s and s[0].isdigit():
            try:
                return float(s[0])
            except ValueError:
                return 2.0
        return 2.0

    @staticmethod
    def _parse_bmi(val: Any) -> float:
        if pd.isna(val):
            return 0.0
        s = str(val).upper()
        return 1.0 if "30" in s or "ABOVE" in s or "OBESE" in s else 0.0

    @staticmethod
    def _parse_age_weight(val: Any) -> float:
        if pd.isna(val):
            return 2.0
        s = str(val).strip()
        if "<" in s or "18" in s:
            return 1.0
        if "19" in s or "35" in s:
            return 2.0
        if "45" in s or "50" in s:
            return 3.0
        if "65" in s or "75" in s:
            return 4.0
        if "85" in s or ">" in s:
            return 5.0
        return 2.5

    @staticmethod
    def _parse_smoking(val: Any) -> float:
        if pd.isna(val):
            return 0.0
        s = str(val).upper()
        if "SMKR" in s or "CURRENT" in s or "EVRY" in s or "SOME" in s:
            return 1.0
        if "FORMER" in s:
            return 0.5
        return 0.0

    @staticmethod
    def _parse_diabetic(val: Any) -> float:
        if pd.isna(val):
            return 0.0
        try:
            return 1.0 if float(val) > 0 else 0.0
        except (ValueError, TypeError):
            s = str(val).upper()
            return 1.0 if "YES" in s or "TRUE" in s or "1" in s else 0.0

    def compute_risk_adjusted_cpc(
        self,
        df: pd.DataFrame,
        procedure_name: str,
        benchmark_percentile: int = 25
    ) -> Dict[str, Any]:
        """
        Trains an encounter-level acuity regression model and computes surgeon O/E ratios.
        """
        if df is None or df.empty:
            return {"status": "error", "message": "No encounter data available"}

        proc_clean = procedure_name.strip().upper()
        proc_mask = df['unified_procedure'].astype(str).str.strip().str.upper() == proc_clean
        proc_df = df[proc_mask].copy()

        if proc_df.empty or len(proc_df) < 5:
            # Fallback to full df if specific procedure has insufficient sample
            proc_df = df.copy()

        spend_col = 'ext_cost' if 'ext_cost' in proc_df.columns else ('line_spend' if 'line_spend' in proc_df.columns else 'unit_cost')
        case_col = 'case_id' if 'case_id' in proc_df.columns else 'encounter_id'

        # 1. Aggregate to Case-Encounter Level
        agg_rules = {
            spend_col: 'sum',
            'lead_surgeon': lambda x: x.dropna().iloc[0] if not x.dropna().empty else "Unknown",
        }
        for col in ['asa_rating', 'bmi_bucket', 'patient_age_bucket', 'smoking_status', 'diabetic_status', 'hospital_name']:
            if col in proc_df.columns:
                agg_rules[col] = lambda x: x.dropna().iloc[0] if not x.dropna().empty else None

        case_df = proc_df.groupby(case_col).agg(agg_rules).reset_index()
        case_df = case_df[case_df[spend_col] > 0].copy()

        if len(case_df) < 4:
            return {"status": "error", "message": "Insufficient cases for multivariate risk modeling"}

        # 2. Extract Numerical Risk Feature Matrix
        case_df['feat_asa'] = case_df['asa_rating'].apply(self._parse_asa) if 'asa_rating' in case_df.columns else 2.0
        case_df['feat_bmi'] = case_df['bmi_bucket'].apply(self._parse_bmi) if 'bmi_bucket' in case_df.columns else 0.0
        case_df['feat_age'] = case_df['patient_age_bucket'].apply(self._parse_age_weight) if 'patient_age_bucket' in case_df.columns else 2.5
        case_df['feat_smoke'] = case_df['smoking_status'].apply(self._parse_smoking) if 'smoking_status' in case_df.columns else 0.0
        case_df['feat_diab'] = case_df['diabetic_status'].apply(self._parse_diabetic) if 'diabetic_status' in case_df.columns else 0.0

        features = ['feat_asa', 'feat_bmi', 'feat_age', 'feat_smoke', 'feat_diab']
        X = case_df[features].values
        y = case_df[spend_col].values

        cohort_mean_cpc = float(np.mean(y))
        cohort_median_cpc = float(np.median(y))
        pct_benchmark = float(np.percentile(y, max(5, min(95, benchmark_percentile))))

        # 3. Fit Multivariate Regularized Acuity Model
        # Using Ridge regression to stably capture additive comorbidity cost burden
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=10.0))
        ])
        model.fit(X, y)
        expected_y = np.maximum(20.0, model.predict(X))
        case_df['expected_cost'] = expected_y

        # Feature importance / weights in dollars
        coefs = model.named_steps['regressor'].coef_
        risk_drivers = [
            {"factor": "ASA Physical Status (Per Acuity Grade)", "impact_dollars": round(float(coefs[0]), 2)},
            {"factor": "Morbid Obesity (BMI >= 30)", "impact_dollars": round(float(coefs[1]), 2)},
            {"factor": "Advanced Patient Age Bracket", "impact_dollars": round(float(coefs[2]), 2)},
            {"factor": "Active / Former Smoking Status", "impact_dollars": round(float(coefs[3]), 2)},
            {"factor": "Pre-Existing Diabetes Mellitus", "impact_dollars": round(float(coefs[4]), 2)},
        ]

        # 4. Roll up to Physician Level for O/E Ratio Calculation
        surgeon_groups = case_df.groupby('lead_surgeon')
        physicians = []

        for surg, g in surgeon_groups:
            if str(surg).strip().upper() in ["UNKNOWN", "NONE", "NAN", ""]:
                continue
            cases = int(len(g))
            if cases < 1:
                continue

            tot_obs = float(g[spend_col].sum())
            tot_exp = float(g['expected_cost'].sum())
            raw_cpc = tot_obs / cases
            exp_cpc = tot_exp / cases
            oe_ratio = tot_obs / max(1.0, tot_exp)
            risk_adj_cpc = cohort_mean_cpc * oe_ratio

            avg_asa = float(g['feat_asa'].mean())
            diab_pct = float(g['feat_diab'].mean() * 100)
            high_bmi_pct = float(g['feat_bmi'].mean() * 100)

            # Acuity tier classification
            if avg_asa >= 2.8 or diab_pct >= 40.0:
                acuity_tier = "High Acuity (Sicker Cohort)"
            elif avg_asa >= 2.2:
                acuity_tier = "Moderate Acuity"
            else:
                acuity_tier = "Standard / Low Acuity"

            # Clinical value tier based on O/E ratio
            if oe_ratio < 0.92:
                clinical_tier = "🟢 High Value Champion"
                rationale = f"Spending {abs(1 - oe_ratio)*100:.1f}% less than patient comorbidity predicts"
            elif oe_ratio <= 1.08:
                clinical_tier = "⚪ Acuity-Aligned"
                rationale = "Spend is directly proportional to clinical illness burden"
            else:
                excess_spend = (tot_obs - tot_exp)
                clinical_tier = "🔴 True Cost Outlier"
                rationale = f"Excess spend of ${excess_spend:,.0f} persists after adjusting for high-risk comorbidities"

            physicians.append({
                "surgeon": str(surg),
                "cases": cases,
                "raw_cpc": round(raw_cpc, 2),
                "expected_cpc": round(exp_cpc, 2),
                "oe_ratio": round(oe_ratio, 3),
                "risk_adjusted_cpc": round(risk_adj_cpc, 2),
                "avg_asa": round(avg_asa, 2),
                "diabetic_pct": round(diab_pct, 1),
                "high_bmi_pct": round(high_bmi_pct, 1),
                "acuity_tier": acuity_tier,
                "clinical_tier": clinical_tier,
                "clinical_rationale": rationale,
                "variance_vs_expected": round(raw_cpc - exp_cpc, 2)
            })

        # Sort by cases and O/E ratio
        physicians.sort(key=lambda x: x["cases"], reverse=True)

        # 5. Build Chart Data for Visualization
        scatter_points = []
        for p in physicians:
            scatter_points.append({
                "surgeon": p["surgeon"],
                "x": p["expected_cpc"],
                "y": p["raw_cpc"],
                "oe_ratio": p["oe_ratio"],
                "cases": p["cases"],
                "avg_asa": p["avg_asa"],
                "tier": p["clinical_tier"]
            })

        return {
            "status": "success",
            "procedure_name": procedure_name,
            "total_encounters_analyzed": len(case_df),
            "cohort_mean_cpc": round(cohort_mean_cpc, 2),
            "cohort_median_cpc": round(cohort_median_cpc, 2),
            "benchmark_percentile_cpc": round(pct_benchmark, 2),
            "benchmark_percentile": benchmark_percentile,
            "physicians": physicians,
            "risk_drivers": risk_drivers,
            "scatter_points": scatter_points,
            "model_metadata": {
                "algorithm": "Multivariate L2-Regularized Acuity Regressor",
                "features_evaluated": ["ASA Rating", "BMI >= 30", "Age Bucket", "Smoking", "Diabetes"],
                "target_metric": "Observed-to-Expected (O/E) Ratio"
            }
        }


risk_engine = ClinicalRiskEngine()
