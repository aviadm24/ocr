import requests
import json
url = 'http://127.0.0.1:8000/ocr_output/'
url2 = 'https://avi-ocr.herokuapp.com/ocr_output/'
files = {'image': open('images/8.jpg', 'rb')}
res = requests.post(url2, files=files)
print('res: ', res)
string = res.text
json_obj = json.loads(string)

print(json_obj)
