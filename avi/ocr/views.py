from django.shortcuts import render
import pytesseract
from PIL import Image
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt


def home(request):
    return render(request, template_name='ocr/home.html')


def plain_ocr(filename):
    text = pytesseract.image_to_string(Image.open(filename), lang='heb')
    for i in text.split('/n'):
        print(i)
    with open('after_clean7.txt', 'w', encoding='utf8') as f:
        f.write(text)
    return text


# https://stackoverflow.com/questions/53363547/how-to-deploy-pytesseract-to-heroku
@csrf_exempt
def image_upload(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save('ocr/static/images/'+myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        text = plain_ocr(uploaded_file_url)
        uploaded_file_url = '/'.join(fs.url(filename).split('/')[2:])
        # print('ocr text: ', text)
        return render(request, 'ocr/image_upload.html', {
            'text': text,
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'ocr/image_upload.html')

