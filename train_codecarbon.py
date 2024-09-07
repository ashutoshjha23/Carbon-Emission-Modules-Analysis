import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from codecarbon import EmissionsTracker
from cvt_model import load_vit_model
import pandas as pd
import time

model = load_vit_model(pretrained=True)
print(model)

tracker_codecarbon = EmissionsTracker()
tracker_codecarbon.start()

model.train()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
dummy_input = torch.randn(16, 3, 224, 224)
dummy_target = torch.randint(0, 1000, (16,))

start_time = time.time()
max_time = 10  # runtime in seconds

while True:
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = F.cross_entropy(output, dummy_target)
    loss.backward()
    optimizer.step()
    elapsed_time = time.time() - start_time
    if elapsed_time >= max_time:
        break

tracker_codecarbon.stop()
emissions_data_codecarbon = tracker_codecarbon.final_emission_model

data_codecarbon = {
    'Row': [1],
    'Metric': ['Carbon emissions (kg CO2eq)'],
    'Value': [round(emissions_data_codecarbon['emissions'], 3)]
}
df_codecarbon = pd.DataFrame(data_codecarbon)

file_path = 'emissions.csv'

# Write to CSV
try:
    df_codecarbon.to_csv(file_path, index=False)
    print(f"Data has been appended to {file_path}")
except Exception as e:
    print(f"An error occurred while writing to CSV: {e}")
