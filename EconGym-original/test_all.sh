#!/bin/bash

# 所有 YAML 场景文件
scenes=(
  "delayed_retirement"
  "personal_pension"
  "population_aging"
  "pension_gap"
  "pension_across_countries"
  "consumption_tax"
  "estate_tax"
  "universal_basic_income"
  "optimal_tax"
  "wealth_tax"
  "negative_interest"
  "inflation_control"
  "QE"
  "optimal_monetary"
  "dbl_government"
  "technology"
  "monopoly"
  "oligopoly"
  "monopolistic_competition"
  "work_hard"
  "age_consumption"
  "asset_allocation"
  "work_life_well_being"
  "over_leveraged_consumption"
  "market_type"
)

# 错误日志文件
error_log="errors.log"
> "$error_log"  # 清空旧的 log

for scene in "${scenes[@]}"; do
    echo "Running scene: $scene ..."
    python main.py --problem_scene "$scene" > "logs/${scene}.out" 2> "logs/${scene}.err"

    if [ $? -ne 0 ]; then
        echo "❌ Scene $scene failed. See logs/${scene}.err" | tee -a "$error_log"
    else
        echo "✅ Scene $scene finished successfully."
    fi
done

echo "Done. Errors summarized in $error_log"
