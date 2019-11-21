from django.shortcuts import render
import pytesseract
from PIL import Image
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
import difflib
import os
from dateutil.parser import parse
from django.conf import settings

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


#  https://guides.gdpicture.com/content/Affecting%20Tesseract%20OCR%20engine%20with%20special%20parameters.html
INVOICE_WORD_LIST = ["קבלה", "חשבונית"]


def home(request):
    return render(request, template_name='ocr/home.html')


def plain_ocr(filename):
    text = pytesseract.image_to_string(Image.open(filename), lang='eng+heb')
    # for i in text.split('/n'):
        # print(i)
    # with open('after_clean7.txt', 'w', encoding='utf8') as f:
    #     f.write(text)
    return text


def close_match(text):
    answers = []
    word_list = text.split()
    # print(word_list)
    for invoice_kind in INVOICE_WORD_LIST:
        found = difflib.get_close_matches(invoice_kind, word_list)
        # print('difflib: ', found)
        if found:
            found_indexs = [i for i, val in enumerate(word_list) if val == found[0]]
            for indx in found_indexs:
                for word in word_list[indx: indx + 5]:
                    # print('invoice: ', word)
                    if any(char.isdigit() for char in word):
                        if is_date(word):
                            answers.append(('קשור לתאריך '+invoice_kind, word))
                        elif len(word) > 4:
                            answers.append((invoice_kind, word))
    return answers

    # return difflib.get_close_matches(INVOICE_WORD_LIST, word_list)


# https://stackoverflow.com/questions/53363547/how-to-deploy-pytesseract-to-heroku
@csrf_exempt
def image_upload(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        cpath = os.getcwd()
        image_path = os.path.join(cpath, 'ocr/static/images/')
        # for filename in os.listdir(image_path):
        #     os.remove(os.path.join(image_path, filename))
        fs = FileSystemStorage()
        filename = fs.save('ocr/static/images/'+myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        print(uploaded_file_url)
        text = plain_ocr(uploaded_file_url)
        cheshbonit = close_match(text)
        uploaded_file_url = '/'.join(fs.url(filename).split('/')[2:])
        print(uploaded_file_url)
        return render(request, 'ocr/image_upload.html', {
            'text': text,
            'cheshbonit': str(cheshbonit),
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'ocr/image_upload.html')

