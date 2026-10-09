import os
from tacoreader import load

os.makedirs("data/sen2naipv2", exist_ok=True)
print("Connecting to Hugging Face to load SEN2NAIPv2 subset...")

# Load Cloud-Optimized dataset subset for UNet training
dataset = load("tacofoundation:sen2naipv2-unet")
print("Dataset loaded successfully. Ready for patch extraction.")