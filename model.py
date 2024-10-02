import json

def load_model_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['models']

carbon_intensity_gCO2_per_kWh = 90  
def calculate_carbon_emissions(models, carbon_intensity):
    emissions = {}
    for model, consumption in models.items():
        emissions[model] = (consumption * carbon_intensity) / 1000 
    return emissions

if __name__ == "__main__":
    models = load_model_data('models_data.json')

    carbon_emissions = calculate_carbon_emissions(models, carbon_intensity_gCO2_per_kWh)

    for model, emission in carbon_emissions.items():
        print(f"{model}: {emission:.4f} kg CO2")
