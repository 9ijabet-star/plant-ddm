from tensorflow.keras.models import load_model

# Load your trained model
model_path = "plant_disease_model_best.keras"
model = load_model(model_path)

# Display model architecture
model.summary()
