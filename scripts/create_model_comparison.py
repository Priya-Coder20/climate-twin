import os
import pandas as pd

OUTPUT_DIR = r"outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = [
    ["Persistence", 0.7511, 1.0294, None],
    ["CNN", 1.3231, 1.7242, 0.3858],
    ["ConvLSTM", 0.8836, 1.1606, 0.7217],
    ["XGBoost", 0.7429, 0.9939, 0.7959],
]

df = pd.DataFrame(
    results,
    columns=["Model", "MAE_C", "RMSE_C", "R2"]
)

output_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

df.to_csv(output_path, index=False)

print(df.to_string(index=False))
print(f"\nSaved: {output_path}")