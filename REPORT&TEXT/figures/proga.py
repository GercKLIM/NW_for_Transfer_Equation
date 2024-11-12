from random import choices, choice, randint
import openpyxl


all_fake_sms, all_true_sms, all_ksi = [], [], []

def make_true_sms():
    for _ in range(100):
        all_true_sms.append([choice(['0', '1']) for _ in range(5)])


def make_fake_sms():
    for sms in all_true_sms:
        fake_sms = []
        for i in range(5):
            p = randint(0, 100)
            if sms[i] == '0' and p <= 10:
                fake_sms.append('1')
            elif sms[i] == '1' and p <= 20:
                fake_sms.append('0')
            else:
                fake_sms.append(sms[i])
        all_fake_sms.append(fake_sms)


def analize_ksi():
    ksi_number = [0] * 6
    for i in range(100):
        kol_true = 0
        for j in range(5):
            if all_true_sms[i][j] == all_fake_sms[i][j]:
                kol_true += 1

        all_ksi.append(kol_true)
        ksi_number[kol_true] += 1

    print(*ksi_number)
