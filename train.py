import time
import pandas as pd
import psutil

CARBON_INTENSITY = 0.9 

def estimate_power(cpu_usage_percent):

    max_cpu_power = 45  
    return (cpu_usage_percent / 100) * max_cpu_power  
def calculate_emissions(energy_joules, carbon_intensity):
    energy_kwh = energy_joules / (3.6 * 10**6) 
    emissions_kg = energy_kwh * carbon_intensity  
    return emissions_kg

data = []

row_number = 1

for _ in range(10):  
    cpu_usage_percent = psutil.cpu_percent(interval=1)  
    battery = psutil.sensors_battery()  

    power_watts = estimate_power(cpu_usage_percent)
  
    energy_joules = power_watts * 1  

    emissions_kg = calculate_emissions(energy_joules, CARBON_INTENSITY)

    data.append({
        'row_number': row_number,  
        'timestamp': time.time(),
        'cpu_usage_percent': cpu_usage_percent,
        'power_watts': power_watts,
        'energy_joules': energy_joules,
        'emissions_kg': emissions_kg,
        'battery_percent': battery.percent if battery else None,
        'power_plugged': battery.power_plugged if battery else None
    })

    row_number += 1
    
    time.sleep(5) 

df = pd.DataFrame(data)

df.to_csv('energy_emissions_data_with_numbers.csv', index=False)

print("Data saved to energy_emissions_data_with_numbers.csv")
