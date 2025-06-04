
class Ot:
    pass


class Mila:
    pass


class Parasha:
    def __init__(self):
        self.words = []


vhy = Parasha()


with open('../parasha_letters/vehaya_ke_yeviacha.txt', 'r') as f:
    for line in f.readlines():
        vhy.words += line.split()
print(vhy.words[0:10])

