import requests
import json
url = 'http://127.0.0.1:8000/ocr_output/'
files = {'image': open('images/6.jpg', 'rb')}
res = requests.post(url, files=files)
string = res.text
print(string)
json_obj = json.loads(string)

print(json_obj) # prints the string with 'source_name' key
# print(json.loads(res.text))
