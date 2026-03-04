import os
import json
import pandas as pd

# --- CONFIG: List of COCO JSON directories ---
folders = [
    r"C:\Users\mjiafeng.DSONET\Desktop\SD6106 Capstone Project_2\2b. Convert_to_YOLO_readable_txt\coco_jsons"
]

# --- Output Excel file ---
output_excel = r"C:\Users\mjiafeng.DSONET\Desktop\SD6106 Capstone Project_2\COCO_Class_Counts_All.xlsx"

# --- Dictionary to store all results for summary ---
all_results = {}

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    for folder in folders:
        folder_name = os.path.basename(folder)
        class_counts = {}

        # Loop through all JSON files in folder
        for filename in os.listdir(folder):
            if filename.lower().endswith(".json"):
                file_path = os.path.join(folder, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Map category_id → category_name
                categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

                # Count each annotation
                for ann in data.get("annotations", []):
                    cat_id = ann["category_id"]
                    cat_name = categories.get(cat_id, f"Unknown({cat_id})")
                    class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

        # Convert to DataFrame and sort
        df = pd.DataFrame(list(class_counts.items()), columns=["Class Name", "Count"])
        df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)

        # Write to Excel (each folder = 1 sheet)
        df.to_excel(writer, sheet_name=folder_name[:31], index=False)

        # Store for summary
        all_results[folder_name] = df

    # --- Combine all results for summary ---
    summary_df = None
    for folder_name, df in all_results.items():
        df = df.rename(columns={"Count": folder_name})
        if summary_df is None:
            summary_df = df
        else:
            summary_df = pd.merge(summary_df, df, on="Class Name", how="outer")

    # Fill missing values, compute total
    summary_df = summary_df.fillna(0)
    summary_df["Total"] = summary_df.drop(columns=["Class Name"]).sum(axis=1)
    summary_df = summary_df.sort_values(by="Total", ascending=False).reset_index(drop=True)

    # Write summary sheet
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

# --- Display summary in Spyder console ---
print("\n✅ COCO Class Counts Summary Across All Folders:\n")
print(summary_df.to_string(index=False))

print(f"\n✅ Results saved to Excel:\n{output_excel}\n")
