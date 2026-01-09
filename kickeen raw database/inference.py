import os
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="fWc7y1W3iEbxUWbboCgv"
)

# Folder containing your 187 images
image_folder = "C:/Users/kreis/Documents/KICKEEN DATABASE"

# Loop through all image files
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)
        result = client.run_workflow(
            workspace_name="kickeen",
            workflow_id="find-goalposts-balls-and-players",
            images={"image": image_path},
            use_cache=True
        )
        print(f"Processed {filename}:")
        print(result)