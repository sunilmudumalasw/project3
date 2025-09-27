"""
Quick API test
Author: Sunil Mudumala
Date: 23/Sep/2023
"""

import requests
import json

# Test GET endpoint
print("Testing GET endpoint...")
response = requests.get("http://127.0.0.1:8000/")
print(f"GET Status: {response.status_code}")
print(f"GET Response: {response.json()}")
print()

# Test POST endpoint with CORRECT URL
print("Testing POST endpoint...")
url = "http://127.0.0.1:8000/predictions/"  # Note the trailing slash!

# Use the exact example from your Pydantic model
data = {
    "age": 35,
    "workclass": "Private",
    "fnlgt": 7777,
    "education": "Some-college",
    "education_num": 10,
    "marital_status": "Never-married",
    "occupation": "Armed-Forces",
    "relationship": "Not-in-family",
    "race": "Other",
    "sex": "Male",
    "capital_gain": 2000,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "Holand-Netherlands"
}

try:
    response = requests.post(url, json=data)
    print(f"POST Status: {response.status_code}")
    print(f"POST Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")