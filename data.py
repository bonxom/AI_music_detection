import kagglehub

# Download latest version
path = kagglehub.dataset_download("awsaf49/sonics-dataset")
path
print("Path to dataset files:", path) 