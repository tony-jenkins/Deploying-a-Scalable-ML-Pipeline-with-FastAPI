import requests

# Base URL for local API
BASE_URL = "http://127.0.0.1:8000"

response_get = requests.get(f"{BASE_URL}/")

print("Status Code:", response_get.status_code)
print("Result:", response_get.json()["message"])

sample_data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

response_post = requests.post(f"{BASE_URL}/data/", json=sample_data)

print("Status Code:", response_post.status_code)
print("Result:", response_post.json()["result"])
