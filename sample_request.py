"""
Request to the Census API
Author: Sunil Mudumala
Date: 26 Sep 2025
"""

import requests
import json
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# local_url = "http://127.0.0.1:7000/predictions/"
cloud_url = "https://census-mldevops.onrender.com/predictions/"

sample = {
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

response = requests.post(cloud_url, data=json.dumps(sample))

logger.info(f"sending sample request to {cloud_url}")
logger.info(f"Status code received -> {response.status_code}")
logger.info(f"Sample prediction -> {response.json()}")