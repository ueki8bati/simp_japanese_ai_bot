from django.shortcuts import render
from django.http import HttpResponse
import json
from django.views.decorators.csrf import csrf_exempt

from utils import message_creater, drive_ocr, output_fl

from line_bot_ai.line_message import LineMessage, GetTmpImage, send_button_template

import os

SAVE_MODEL_NAME = "snow_matcha"

# NGワードを追加する関数
def add_ng_phrase(keyword, phrase, file_path='/home/yuki_ueda/AI_line_bot/line_bot/utils/ng_words.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            ng_words = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):  # JSONDecodeErrorも追加
        ng_words = {}

    if keyword in ng_words:
        if phrase not in ng_words[keyword]:
            ng_words[keyword].append(phrase)
    else:
        ng_words[keyword] = [phrase]

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(ng_words, file, ensure_ascii=False, indent=4)

def send_text_message(text):
    return {"type": "text", "text": text}


@csrf_exempt
def index(request):
    if request.method == 'POST':
        request = json.loads(request.body.decode('utf-8'))
        data = request['events'][0]

        with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/i_views.txt', 'r', encoding='utf-8') as file:
            get_same_number = file.readline().strip()  # .strip() で前後の空白や改行を削除

        if get_same_number:
            i = int(get_same_number)
        else:
            # get_same_number が空の場合の処理
            i = 0  # 例えば、デフォルト値を0にするなど
        
        with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/3_views.txt', 'r', encoding='utf-8') as file:
            get_same_number_s = file.readline().strip()  # .strip() で前後の空白や改行を削除

        if get_same_number_s:
            s = int(get_same_number_s)
        else:
            # get_same_number が空の場合の処理
            s = 0  # 例えば、デフォルト値を0にするなど
        

        if data['type'] == 'postback':  # postbackイベントが発生した場合
            # 「いいえ」ボタンが押されたときの処理
            if data['postback']['data'] == 'action=no':
                # output_llm.txtからキーワードを取得
                with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/output_llm.txt', 'r', encoding='utf-8') as f:
                    keyword = f.read().strip()
                
                # snow_matcha_test_tgt.txtからNGワードを取得
                    with open("/home/yuki_ueda/AI_line_bot/line_bot/utils/" + SAVE_MODEL_NAME + "_test_tgt.txt", 'r', encoding='utf-8') as f:
                        ng_phrase = f.read().strip()
                        keyword = keyword.rstrip('。')
                        ng_phrase = ng_phrase.rstrip('。')
                        add_ng_phrase(keyword, ng_phrase)
                        complete_message = f"キーワード:{keyword}\nNG:{ng_phrase}\n登録完了"


                # 完了メッセージをLINEに返信
                line_message = LineMessage(messages=[send_text_message(complete_message)])
                line_message.reply(data['replyToken'])
            else:
                # キーワードとNGワードの登録完了メッセージを作成
                complete_message = "OK!"

                # 完了メッセージをLINEに返信
                line_message = LineMessage(messages=[send_text_message(complete_message)])
                line_message.reply(data['replyToken'])
                
            return HttpResponse("ok")


        message = data['message']
        reply_token = data['replyToken']
        if message['type'] == 'image':
            # 画像メッセージの処理
            with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/get_img_or_text.txt', 'w', encoding='utf-8') as file:
                    img_or_text = 0
                    file.write(str(img_or_text))
            tmp_img = GetTmpImage(message['id'])
            service = drive_ocr.get_service()
            ocr_output_path = drive_ocr.read_ocr(service, tmp_img, 'ja')
            output_fl.write_ocr_to_file(ocr_output_path)
            if i == 1:
                message_btn = send_button_template()
                line_message = LineMessage(messages=[message_creater.create_single_text_message(), message_btn])
                if s == 1:
                    line_message = LineMessage(messages=[message_creater.create_single_text_message(), message_btn, message_creater.create_all_text_message()])
            elif s == 1:
                line_message = LineMessage(message_creater.create_all_text_message())
            else:
                line_message = LineMessage(message_creater.create_single_text_message())

        else:
            with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/get_img_or_text.txt', 'w', encoding='utf-8') as file:
                    img_or_text = 1
                    file.write(str(img_or_text))
            if message['text'] == "settings":
                i = 1
                with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/i_views.txt', 'w', encoding='utf-8') as file:
                    file.write(str(i))
                    complete_message = "設定モード:開始"
                    line_message = LineMessage(messages=[send_text_message(complete_message)])
            elif message['text'] == "settingf":
                i = 0
                with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/i_views.txt', 'w', encoding='utf-8') as file:
                    file.write(str(i))
                    complete_message = "設定モード:終了"
                    line_message = LineMessage(messages=[send_text_message(complete_message)])
            elif message['text'] == "candidates":
                s = 1
                with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/3_views.txt', 'w', encoding='utf-8') as file:
                    file.write(str(s))
                    complete_message = "候補文表示:開始"
                    line_message = LineMessage(messages=[send_text_message(complete_message)])
            elif message['text'] == "candidatef":
                s = 0
                with open('/home/yuki_ueda/AI_line_bot/line_bot/utils/3_views.txt', 'w', encoding='utf-8') as file:
                    file.write(str(s))
                    complete_message = "候補文表示:終了"
                    line_message = LineMessage(messages=[send_text_message(complete_message)])

            else:
                output_fl.write_text_to_file(message['text'])
                if i == 1:
                    message_btn = send_button_template()
                    line_message = LineMessage(messages=[message_creater.create_single_text_message(), message_btn])
                    if s == 1:
                        line_message = LineMessage(messages=[message_creater.create_single_text_message(), message_btn, message_creater.create_all_text_message()])
                elif s == 1:
                    line_message = LineMessage(message_creater.create_all_text_message())
                else:
                    line_message = LineMessage(message_creater.create_single_text_message())
        line_message.reply(reply_token)
        return HttpResponse("ok")





