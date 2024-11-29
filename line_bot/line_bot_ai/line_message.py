from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import urllib.request
import json
import os
from django.conf import settings
from linebot import (LineBotApi, WebhookHandler)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage,TemplateSendMessage,PostbackAction,ButtonsTemplate,ImageSendMessage)
from linebot.exceptions import (LineBotApiError, InvalidSignatureError)


REPLY_ENDPOINT_URL = "https://api.line.me/v2/bot/message/reply"
ACCESSTOKEN = 'ltCqn3sc07fHJyJQaTjR2TCexfCRbyhIxTUb6wq/0+AcdDQHsNCTI7a8fFo2tFY+mlRozzOD4T+RvK4te1ouI1eALSXtC3kR2HN7oX0V+k8E5n6NJo54ayL3hPgHk2nWVTs5HgIH7h71PI8tkFBe+gdB04t89/1O/w1cDnyilFU='
line_bot_api = LineBotApi(ACCESSTOKEN)
HEADER = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + ACCESSTOKEN
}

class LineMessage():
    def __init__(self, messages):
        # messagesをリストとして受け取る
        if isinstance(messages, list):
            self.messages = messages
        else:
            self.messages = [messages]  # 単一のメッセージの場合はリストに変換

    def serialize_message(self, message):
        """メッセージオブジェクトを辞書に変換する"""
        if isinstance(message, TemplateSendMessage):
            return {
                'type': 'template',
                'altText': message.alt_text,
                'template': {
                    'type': 'buttons',
                    'text': message.template.text,
                    'actions': [self.serialize_action(action) for action in message.template.actions]
                }
            }
        # 他のメッセージタイプについても必要に応じて処理を追加
        return message

    def serialize_action(self, action):
        """アクションオブジェクトを辞書に変換する"""
        if isinstance(action, PostbackAction):
            return {
                'type': 'postback',
                'label': action.label,
                'data': action.data,
                'displayText': action.display_text
            }
        # 他のアクションタイプについても必要に応じて処理を追加


    def reply(self, reply_token):
        body = {
            'replyToken': reply_token,
            'messages': [self.serialize_message(msg) for msg in self.messages]
        }
        print(body)
        req = urllib.request.Request(REPLY_ENDPOINT_URL, json.dumps(body).encode(), HEADER)
        try:
            with urllib.request.urlopen(req) as res:
                body = res.read()
        except urllib.error.HTTPError as err:
            print(err)
            if err.read():
                error_body = err.read()
                print(f'Error body: {error_body}') 
        except urllib.error.URLError as err:
            print(err.reason)

def GetTmpImage(message_id):
    message_content = line_bot_api.get_message_content(message_id)

    # ファイルの保存パスを設定
    image_path = os.path.join(settings.MEDIA_ROOT, "tmp.jpg")
    
    # 画像を保存
    with open(image_path, 'wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)

    return image_path

def send_button_template():
    button_template = TemplateSendMessage(
        alt_text="選択してください",
        template=ButtonsTemplate(
            text="正しくやさしくなっていますか？",
            actions=[
                PostbackAction(label="はい", data="action=yes"),
                PostbackAction(label="いいえ", data="action=no"),
                PostbackAction(label="わからない", data="action=unknown")
            ]
        )
    )
    return button_template



