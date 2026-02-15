import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "from": "Sao Paulo (SP)",
    "to": "Rio de Janeiro (RJ)",
    "flightType": "economic",
    "agency": "FlyingDrops",
    "weekday_num": 3,
    "month": 5,
    "year": 2026
}

response = requests.post(url, data=data)
print(response.json())
