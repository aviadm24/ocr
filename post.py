import requests
import json
from PIL import Image
url = 'http://127.0.0.1:8000/ocr_output/'
url2 = 'https://avi-ocr.herokuapp.com/ocr_output/'
files = {'image': open('images/7.jpg', 'rb')}
res = requests.post(url, files=files)
string = res.text
json_obj = json.loads(string)

print(json_obj)
