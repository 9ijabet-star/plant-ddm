import json

# Define the exact classes your model was trained with
class_indices = {"Diseased": 0, "Healthy": 1, "Non_Plant_images": 2}

# Save to JSON file
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

print("✅ class_indices.json saved successfully!")
