import requests as requests
import datetime

today = str(datetime.datetime.today())
id = "yourID"
token = 'yourToken'


def sendMessage():
    send = "Se detectó movimiento. "

    url = "https://api.telegram.org/bot" + token + "/sendMessage"
    params = {
        'chat_id': id,
        'text': send
    }
    requests.post(url, params=params)


def sendVideo():
    files = {'video': open('videoSalida.avi', 'rb')}
    requests.post("https://api.telegram.org/bot{}/sendVideo?chat_id={}&caption={}".format(token, id, today),
                  files=files)
