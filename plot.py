import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('energy_emissions_data_with_numbers.csv')
df.columns = df.columns.str.strip()

print("Cleaned Columns in CSV file:", df.columns)

row_col = 'row_number'
emissions_col = 'emissions_kg'

if row_col in df.columns and emissions_col in df.columns:
    plt.style.use('ggplot')  
    plt.figure(figsize=(12, 8))

    bars = plt.bar(df[row_col], df[emissions_col], color='coral', edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

    plt.xlabel('Row Number', fontsize=14)
    plt.ylabel('Emissions (kg CO2)', fontsize=14)
    plt.title('Carbon Emissions by Row Number', fontsize=16)

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  
    plt.savefig('emissions_by_row_number.png')
    plt.show()
else:
    print(f"Columns '{row_col}' or '{emissions_col}' not found in CSV file.")
