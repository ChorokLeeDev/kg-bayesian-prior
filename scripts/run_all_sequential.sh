#!/bin/bash
# Sequential experiment runner - survives session disconnects
# Each experiment runs one at a time to avoid OOM (3.9GB RAM)

cd /sessions/determined-happy-galileo/mnt/kg-bayesian-prior
LOG="outputs/sequential_runner.log"

echo "=== Sequential Runner Started: $(date) ===" >> $LOG

# 1. Margin loss 30-epoch ablation
echo "[$(date)] Starting margin_loss_ablation_30epoch..." >> $LOG
python scripts/margin_loss_ablation_30epoch.py >> outputs/margin_loss_30epoch.log 2>&1
echo "[$(date)] Margin loss finished with exit code $?" >> $LOG

# 2. Baseline+Coverage ablation
echo "[$(date)] Starting baseline_coverage ablation..." >> $LOG
python scripts/run_baseline_coverage_ablation.py >> outputs/baseline_coverage_30epoch.log 2>&1
echo "[$(date)] Baseline+Coverage finished with exit code $?" >> $LOG

# 3. GDELT pipeline
echo "[$(date)] Starting GDELT pipeline..." >> $LOG
python scripts/gdelt_pipeline.py >> outputs/gdelt_pipeline.log 2>&1
echo "[$(date)] GDELT finished with exit code $?" >> $LOG

# 4. R-GCN/CompGCN
echo "[$(date)] Starting R-GCN/CompGCN..." >> $LOG
python scripts/rgcn_compgcn_experiment.py >> outputs/rgcn_compgcn.log 2>&1
echo "[$(date)] R-GCN/CompGCN finished with exit code $?" >> $LOG

echo "=== All experiments done: $(date) ===" >> $LOG
