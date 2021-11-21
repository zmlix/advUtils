import json
import requests
import base64
import os
from PIL import Image


class GN():
    def __init__(self, project) -> None:
        self.project = project
        self.session = requests.Session()
        self.init()
        self.postUrl = self.baseUrl + self.project

    def init(self, baseUrl="http://23.105.196.211:9999/post/"):
        self.baseUrl = baseUrl
        self.postUrl = self.baseUrl + self.project

    def info(self):
        return f'project: {self.project}'

    def imgTobase64(self, img):
        im = Image.fromarray(img)
        im.save("tmp.jpg")
        save_img = open('./tmp.jpg', 'rb')
        t = base64.b64encode(save_img.read())
        save_img.close()
        os.remove('./tmp.jpg')
        return 'data:image/jpg;base64,' + str(t)[2:-1]

    def text(self, context, level=0):
        send_json = {
            "project": self.project,
            "type": "text",
            "level": level,
            "context": context
        }
        self.session.post(self.postUrl, json=send_json)

    def img(self, img, level=0):
        send_json = {
            "project": self.project,
            "type": "img",
            "level": level,
            "context": self.imgTobase64(img)
        }

        self.session.post(self.postUrl, json=send_json)

    def mix(self, msgList, level=0):

        messages = []

        for type_, msg in msgList:
            if type_ == 'text':
                messages.append({'type': type_, 'context': msg})
            if type_ == 'img':
                messages.append({
                    'type': type_,
                    'context': self.imgTobase64(msg)
                })

        send_json = {
            "project": self.project,
            "type": "mix",
            "level": level,
            "context": json.dumps({'msg': messages})
        }
        self.session.post(self.postUrl, json=send_json)
