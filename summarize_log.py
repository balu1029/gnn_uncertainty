import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("logs/evi_20241129_140558/eval.csv")

metrics = [
    "Energy Correlation In Distribution Energies",
    "Energy Correlation In Distribution Forces",
    # "Force Correlation In Distribution Forces",
    # "Force Correlation In Distribution Energy",
    "Energy Correlation Out Distribution Energy",
    "Energy Correlation Out Distribution Forces",
    # "Force Correlation Out Distribution Energy",
    # "Force Correlation Out Distribution Forces",
    # "Energy R2 Score In Distribution",
    # "Forces R2 Score In Distribution",
    # "Energy R2 Score Out Distribution",
    # "Force R2 Score Out Distribution",
]

for metric in metrics:
    column = df[metric]  # Change 'your_column_name' to the column you want

    # Calculate the mean and standard deviation
    mean_value = column.mean()
    std_value = column.std()

    print(f"{metric} --- Mean: {mean_value}, Standard Deviation: {std_value}")
