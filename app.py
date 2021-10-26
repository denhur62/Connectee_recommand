from flask import Flask, jsonify, abort, request
from train import FTtrain, LDAtrain, VAEtrain
import Fasttext
import LDA
import predict
from apscheduler.schedulers.background import BackgroundScheduler

# 3 time learning
def backgroundScheduler():
    FTtrain()
    LDAtrain()
    VAEtrain()

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(backgroundScheduler, 'interval', minutes=180)
scheduler.start()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


def backgroundScheduler():
    FTtrain()
    LDAtrain()


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(backgroundScheduler, 'interval', minutes=180)
scheduler.start()
application = Flask(__name__)
application.config['JSON_AS_ASCII'] = False

# test url
@application.route('/')  
def index():
    return 'hello world'

# click-emotion and topic add
@application.route('/click-emotion/<userId>/<diaryId>')
def click_emotion(userId=None, diaryId=None):
    LDA.emotion_click(userId, diaryId)
    return '', 202

# click-diary and topic add
@application.route('/click-diary/<userId>/<diaryId>')
def click_diary(userId=None, diaryId=None):
    LDA.diary_click(userId, diaryId)
    return '', 202

# recommand
@application.route('/recommand/<userId>')
def recommand_diary(userId):
    result = predict.recommand_diary(userId)
    return jsonify({'diaries': result})

# search group
@application.route('/group-recommand', methods=['POST'])
def search_group():
    params = request.get_json()
    result = Fasttext.group_recommand(params['search'])
    return jsonify({'groups': result})

# created diary add LDA,Fasttext vector
@application.route('/diary/create', methods=['POST'])
def insert_vector():
    params = request.get_json()
    predict.insert_vector(params['diary_id'],
                          params['title'], params['content'])
    return '', 202

# white list
# @application.before_request
# def limit_remote_addr():
#     if request.remote_addr != '127.0.1.1':
#         abort(403)  # Forbidden


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000, debug=True)
