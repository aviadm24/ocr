from django.shortcuts import render
import pytesseract
from PIL import Image
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
import difflib
#  https://guides.gdpicture.com/content/Affecting%20Tesseract%20OCR%20engine%20with%20special%20parameters.html

def home(request):
    return render(request, template_name='ocr/home.html')


def plain_ocr(filename):
    text = pytesseract.image_to_string(Image.open(filename), lang='eng+heb')
    # for i in text.split('/n'):
        # print(i)
    with open('after_clean7.txt', 'w', encoding='utf8') as f:
        f.write(text)
    return text


def close_match(text):
    return(difflib.get_close_matches('חשבונית', text.split(' ')))

# https://stackoverflow.com/questions/53363547/how-to-deploy-pytesseract-to-heroku
@csrf_exempt
def image_upload(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save('ocr/static/images/'+myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        text = plain_ocr(uploaded_file_url)
        cheshbonit = close_match(text)
        uploaded_file_url = '/'.join(fs.url(filename).split('/')[2:])
        print('ocr text: ', cheshbonit)
        return render(request, 'ocr/image_upload.html', {
            'text': text,
            'cheshbonit': cheshbonit,
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'ocr/image_upload.html')

