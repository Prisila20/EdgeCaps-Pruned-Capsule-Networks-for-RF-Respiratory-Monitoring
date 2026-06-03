import os
import json
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your wandb runs
base_path = "./wandb"


# List to collect results
records = []

# Loop through all run folders
for run_dir in os.listdir(base_path):
    run_path = os.path.join(base_path, run_dir, "files")
    if not os.path.isdir(run_path):
        continue
    
    run_data = {"run": run_dir}
    
    # --- Extract from wandb-metadata.json ---
    meta_path = os.path.join(run_path, "wandb-metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            args = meta.get("args", [])
            for arg in args:
                if "--temperature" in arg:
                    try:
                        run_data["temperature"] = float(arg.split("=")[-1])
                    except:
                        pass
                if "--alpha" in arg:
                    try:
                        run_data["alpha"] = float(arg.split("=")[-1])
                    except:
                        pass
                if "--epochs" in arg:
                    try:
                        run_data["epochs"] = float(arg.split("=")[-1])
                    except:
                        pass
                if "--learning_rate" in arg:
                    try:
                        run_data["learning_rate"] = float(arg.split("=")[-1])
                    except:
                        pass     
                if "--batch_size" in arg:
                    try:
                        run_data["batch_size"] = float(arg.split("=")[-1])
                    except:
                        pass                                            

    
    # --- Extract from wandb-summary.json ---
    summary_path = os.path.join(run_path, "wandb-summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            try:
                summary = json.load(f)
                run_data["teacher_acc"] = summary.get("teacher_val_accuracy")
                run_data["student_acc"] = summary.get("distill_val_accuracy")
            except:
                pass
    
    # Save only if key fields exist
    if "temperature" in run_data and "alpha" in run_data and "learning_rate" in run_data and "epochs" in run_data and "batch_size" in run_data:
        records.append(run_data)

# Convert to DataFrame
df = pd.DataFrame(records)

max_student = df["student_acc"].max()
max_teacher = df["teacher_acc"].max()

print(f"Max Student Accuracy: {max_student:.2f}")
print(f"Max Teacher Accuracy: {max_teacher:.2f}")

best_student_idx = df["student_acc"].idxmax()
best_student_row = df.loc[best_student_idx]

max_student = best_student_row["student_acc"]
best_alpha = best_student_row["alpha"]
best_temp = best_student_row["temperature"]

print(f"Max Student Accuracy: {max_student:.2f}% (alpha={best_alpha}, T={best_temp})")
print(f"Max Teacher Accuracy: {max_teacher:.2f}%")


# Save results
out_csv = os.path.join(base_path, "kd_results_dist.csv")
df.to_csv(out_csv, index=False)

exit()

print("Extraction complete! Results saved to:", out_csv)
print(df.head())


plt.figure(figsize=(8, 6))
for alpha in sorted(df["alpha"].unique()):
    subset = df[df["alpha"] == alpha]
    plt.plot(subset["temperature"], subset["student_acc"], marker="o", label=f"alpha={alpha}")
plt.xlabel("Temperature (T)")
plt.ylabel("Student Accuracy (%)")
plt.title("Effect of Temperature on Student Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 2. Line Plot: Accuracy vs Alpha ---
plt.figure(figsize=(8, 6))
for temp in sorted(df["temperature"].unique()):
    subset = df[df["temperature"] == temp]
    plt.plot(subset["alpha"], subset["student_acc"], marker="s", label=f"T={temp}")
plt.xlabel("Alpha (α)")
plt.ylabel("Student Accuracy (%)")
plt.title("Effect of Alpha on Student Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 3. Heatmap: Student Accuracy (Temperature × Alpha) ---
pivot = df.pivot_table(index="alpha", columns="temperature", values="student_acc", aggfunc="mean")
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
plt.xlabel("Temperature (T)")
plt.ylabel("Alpha (α)")
plt.title("Student Accuracy Heatmap (KD Hyperparameters)")
plt.tight_layout()
plt.show()
