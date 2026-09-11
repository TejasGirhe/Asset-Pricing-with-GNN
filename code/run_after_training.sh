#!/bin/bash
set -e
cd "D:\UCB MFE\Sem3\230ZA\230ZA Replication\code"
# Wait for NN3 to finish (or error)
while true; do
  if grep -q "NN3 trained in" nn_benchmarks_run.log 2>/dev/null; then
    echo "NN3 training complete."
    break
  fi
  if grep -q "Traceback" nn_benchmarks_run.log 2>/dev/null; then
    echo "TRAINING FAILED - see nn_benchmarks_run.log"
    exit 1
  fi
  sleep 20
done
echo "=== Both CSVs ==="
ls -la ../data/firm_level_NN2_predictions_ALL_firms.csv ../data/firm_level_NN3_predictions_ALL_firms.csv
echo "=== Running table5_replication.py ==="
python3 table5_replication.py 2>&1 | tee table5_replication_with_nn23.log
echo "=== DONE ==="
