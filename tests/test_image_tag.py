import requests

url = "http://0.0.0.0:8000/tag-image"
file_path = "test_image2.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print("Response:", response.json())
