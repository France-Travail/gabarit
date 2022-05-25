This is the default model upload instructions file, and you'll need to update it with your own instructions.

This file usually explains how to upload a trained model to a storage solution (such as Artifactory).

Each model saving function will copy this file inside its own directory, and replace a placeholder (default: `model_dir_path_identifier`) by the model's directory path (absolute).
