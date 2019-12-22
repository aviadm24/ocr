import pytesseract
import cv2
from pytesseract import Output
import difflib
from dateutil.parser import parse
import re
from statistics import mean


word_list_dict = {
            "INVOICE": ["קבלה", "חשבונית", "אסמכתא", "חשבונית מס", "קבלה מס."],
            "DATE": ["תאריך", "תאריך חשבונית"],
            "ID": ["ח.פ", "ע.מ", "עוסק מורשה", "עוסק מורשה לקוח"],
            "NETO": ['סה"כ', 'לפני מע"מ', 'סה"כ לפני מע"מ', 'סה"כ אחר הנחה', 'סה"כ חייב מע"מ', 'סה"כ מחיר', 'מחיר כולל'],
            "MAAM": ['מע"מ', '17.00', '%', '17'],
            "BROTO": ['סה"כ לתשלום'],
            "TEL": ['טלפון', 'טל', 'ט"ל', 'מספר', 'פלא', 'נייד'],
            "MAIL": ['Email', 'מייל', 'מייל אלקטרוני', '@', 'il', 'gmail', 'yahoo', 'com', 'co'],
            "URL": ['אתר', 'קישור', 'www']
}


def is_num(text):
    # https: // stackoverflow.com / questions / 46238104 / extracting - prices - with-regex
    r = re.compile(r'(\d[\d.,]*)\b')
    for m in re.finditer(r, text):
    # for m in re.finditer(r"[-+]?\d*\.\,\d+|\d+", text):
        print(m)

# is_num(' סנטר א:א.ג בע"מ ח.פ. 515906196 עמ 518906196      8 טנטר 3   סה"כ לתשלום: 52,782.00')


def is_date(string, fuzzy=True):
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def closest_num(text, word, date, email, tel, url):
    s = re.search(word, text)
    wordend = s.end()
    # print(wordend)
    min_dif = 100
    close_num = None
    # https: // stackoverflow.com / questions / 46238104 / extracting - prices - with-regex
    if date:
        r = re.compile(r"(\d[\d.,/-]*)\b")
    elif email:
        # https: // emailregex.com /
        r = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    elif tel:
        r = re.compile(r"(\d[\d-+]*)\b")
    elif url:
        r = re.compile(r"(^www[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")  # not checked
    else:
        r = re.compile(r'(\d[\d.,]*)\b')
    for m in re.finditer(r, text):
        nstart, num = m.start(), m.group(0)
        # print('%02d-%02d: %s' % (m.start(), m.end(), m.group(0)))
        if nstart > wordend:
            dif = nstart - wordend
            if min_dif < dif:
                pass
            else:
                min_dif = dif
                close_num = num
    # print("closest: ", close_num)
    return close_num


def most_common(lst):
    return max(set(lst), key=lst.count)


def close_match(text, answers, cache):
    print('\ntext: ', text)
    date = False
    email = False
    tel = False
    url = False
    for key, word_list in word_list_dict.items():
        if key == "DATE":
            date = True
        if key == "MAIL":
            email = True
        if key == "TEL":
            tel = True
        if key == "URL":
            url = True
        key_ratio = []
        num_list = []
        for word in word_list:
            for index, t in enumerate(text.split()):
                m = difflib.SequenceMatcher(None, word, t)
                r = round(m.quick_ratio(), 3)
                if r > 0.6:
                    num = closest_num(text, t, date, email, tel, url)
                    # print(word, ' - ', num, ' - ', r)
                    if num == None:
                        r = 0
                    else:
                        num_list.append(num)
                    key_ratio.append(r)
        if key_ratio != [] and num_list != []:
            avr = mean(key_ratio)
            common_num = most_common(num_list)
            print('\n key: ', key)
            print("common_num: ", common_num)
            print("avr: ", avr)
            if avr > cache[key]:
                answers[key] = (avr, common_num)
            cache[key] = avr
            # print(key, ' - ', avr, ' - ', text)
    # print('answers: ', answers)
    # return answers


def data(img_file):
    # https://stackoverflow.com/questions/20831612/getting-the-bounding-box-of-the-recognized-words-using-python-tesseract
    img = cv2.imread('images/8.jpg')
    d = pytesseract.image_to_data(img, lang='heb', output_type=Output.DICT)
    n_boxes = len(d['text'])
    # print(d.keys())
    line_dict = {key: [] for key in d['line_num']}
    for i in range(n_boxes):
        block_num = d['block_num'][i]
        line_num = d['line_num'][i]
        text = d['text'][i]
        if text != '':
            line_dict[line_num].append(text)

        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(line_num), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    # print(line_dict)
    answers = {}
    cache = dict.fromkeys(word_list_dict, 0)
    for line_num, line in line_dict.items():
        # print("line number: ", line_num)
        if line != []:
            close_match(' '.join(line), answers, cache)
    print('answers: ', answers)
    return answers

