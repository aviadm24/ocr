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


def is_date(text):
    r = re.compile(r'(^[0-9]+/[0-9]+/[0-9]+$)')
    if re.findall(r, text):
        return True
    else:
        return False


# print(is_date('0/09.2019'))

# not used
# def closest_date(text):
#     r = re.compile(r"(\d[\d.,/-]*)\b")
#     for word in word_list_dict["DATE"]:
#         found = difflib.get_close_matches(word, text.split())
#         if found != []:
#             date = re.findall(r, text)
#             print(date)
#
#
# closest_date('תאריך ‏ : 31.05/2019')


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


# closest_num('סנטר 1א:א.ג בע"מ ח.פ. 515906196 עמ 518906196      8 טנטר 3   סה"כ לתשלום: ', "ח.פ")

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

# close_match(text)


def boxes():
    # https: // stackoverflow.com / questions / 20831612 / getting - the - bounding - box - of - the - recognized - words - using - python - tesseract
    img = cv2.imread('images/1.png')
    height = img.shape[0]
    width = img.shape[1]

    h, w = img.shape[:2]
    cv2.namedWindow('jpg', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('jpg', w, h)
    # cv2.imshow('jpg', c)

    # d = pytesseract.image_to_boxes(img, lang='heb', output_type=Output.DICT)
    d = pytesseract.image_to_boxes(img, lang='heb', output_type=Output.DICT)
    print(d.keys())
    n_boxes = len(d['char'])
    for i in range(n_boxes):
        (text, x1, y2, x2, y1) = (d['char'][i], d['left'][i], d['top'][i], d['right'][i], d['bottom'][i])
        print(text, ' - x1:{}, y2:{}, x2:{}, y1:{} - page:{}'.format(x1, y2, x2, y1, d['page'][i]))
        cv2.rectangle(img, (x1, height-y1), (x2, height-y2), (0, 255, 0), 2)
    cv2.imshow('jpg', img)
    cv2.waitKey(0)
# boxes()


def data():
    # https://stackoverflow.com/questions/20831612/getting-the-bounding-box-of-the-recognized-words-using-python-tesseract
    img = cv2.imread('images/1.png')
    # heb = pytesseract.image_to_data(img, lang='heb', output_type=Output.DICT)
    digit = pytesseract.image_to_data(img, config='digits', output_type=Output.DICT)

    n_boxes = len(digit['text'])
    # print(d.keys())
    line_dict = {key: [] for key in digit['line_num']}
    # line_dict.fromkeys(d['line_num'])
    # line_list = []
    for i in range(n_boxes):
        block_num = digit['block_num'][i]
        line_num = digit['line_num'][i]
        text = digit['text'][i]
        if text != '':
            line_dict[line_num].append(text)

        (x, y, w, h) = (digit['left'][i], digit['top'][i], digit['width'][i], digit['height'][i])
        # print('{} - x1:{}, y2:{}, x2:{}, y1:{} - level:{}, conf:{}, block_num:{}, par_num:{}, line_num:{}, word_num:{}'.
        #       format(d['text'][i], x, y, w, h, d['level'][i], d['conf'][i], block_num, d['par_num'][i],
        #              line_num, d['word_num'][i])
        #       )
        try:
            float(digit['text'][i])
            print(digit['text'][i])
        except:
            pass
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

    height, width = img.shape[:2]
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', width, height)
    cv2.imshow('img', img)
    cv2.waitKey(0)

# data()


def digits():
    img = cv2.imread('images/8.jpg')
    text = pytesseract.image_to_string(img, config='digits')
    print(text)

# digits()


def heb_digit():
    img = cv2.imread('images/7.jpg')
    heb = pytesseract.image_to_data(img, lang='heb', output_type=Output.DICT)
    digit = pytesseract.image_to_data(img, config='digits', output_type=Output.DICT)

    heb_boxes = len(heb['text'])
    digit_boxes = len(digit['text'])
    print(heb_boxes, ' - ', digit_boxes)
    lines = []
    for i in range(heb_boxes):
        for j in range(digit_boxes):

            # (x, y, w, h) = (digit['left'][i], digit['top'][i], digit['width'][i], digit['height'][i])

            text_pixels = (heb['left'][i], heb['top'][i], heb['width'][i], heb['height'][i])
            digit_pixels = (digit['left'][j], digit['top'][j], digit['width'][j], digit['height'][j])
            if digit_pixels == text_pixels:
                text = heb['text'][i]
                num = digit['text'][j]
                if text != '':
                    print('text: ', heb['text'][i], ' - digit: ', digit['text'][j])
                    if is_date(text):
                        lines.append(text)
                    else:
                        try:
                            if float(num):
                                lines.append(num)
                                lines.append('\n')
                        except ValueError:
                            lines.append(text)
    # print(' '.join(lines))

heb_digit()
