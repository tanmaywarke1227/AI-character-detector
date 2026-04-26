from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="D:\ai-character-detector\ai-character-detector\src",
    repo_id="tannmay27/AI-Animated-Character-detection-model",
    repo_type="model"
)