from flask import Flask, jsonify, abort, request
import API
from train import FTtrain, LDAtrain
from apscheduler.schedulers.background import BackgroundScheduler


def backgroundScheduler():
    FTtrain()
    LDAtrain()


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(backgroundScheduler, 'interval', minutes=180)
scheduler.start()
application = Flask(__name__)
application.config['JSON_AS_ASCII'] = False


@application.route('/')  # 접속하는 url
def index():
    return 'hello world'

# 감정 클릭 시 토픽 추가


@application.route('/click-emotion/<userId>/<diaryId>')
def click_emotion(userId=None, diaryId=None):
    API.click_emotion(userId, diaryId)
    return '', 202

# 다이어리 클릭 시 토픽 추가


@application.route('/click-diary/<userId>/<diaryId>')
def click_diary(userId=None, diaryId=None):
    API.click_diary(userId, diaryId)
    return '', 202
# recommand


@application.route('/recommand/<userId>')
def recommand(userId):
    result = API.recommand(userId)
    return jsonify({'diaries': result})

# 다이어리 생성시,수정 시  두 개 벡터 추가 및 토픽 추가 테스트 해야함


@application.route('/diary/create', methods=['POST'])
def vector():
    params = request.get_json()
    API.vector(params['diary_id'], params['title'], params['content'])
    return '', 202

# white list


# @application.before_request
# def limit_remote_addr():
#     if request.remote_addr != '127.0.1.1':
#         abort(403)  # Forbidden


if __name__ == "__main__":
    application.run(debug=True)
    # app.run(host="127.0.0.1", port="5000", debug=True)
