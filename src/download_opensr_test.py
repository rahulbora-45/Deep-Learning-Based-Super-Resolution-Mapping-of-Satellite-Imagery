import opensr_test
import os

os.makedirs("data/opensr_test", exist_ok=True)
print("Downloading opensr-test benchmark datasets...")

# Load benchmark datasets
spot_data = opensr_test.load("spot")
venus_data = opensr_test.load("venus")

print("opensr-test datasets downloaded and cached successfully.")