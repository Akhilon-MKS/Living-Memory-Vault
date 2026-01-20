import requests

url = 'http://localhost:5000/upload'
files = {'files': open('test_memory.txt', 'rb')}
response = requests.post(url, files=files)
print("Status Code:", response.status_code)
print("Response:", response.json())
