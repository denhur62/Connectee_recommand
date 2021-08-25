from flask import Flask, jsonify, abort, request
import API
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')  # 접속하는 url
def index():
    return 'hello world'

# 감정 클릭 시 토픽 추가


@app.route('/click-emotion/<userId>/<diaryId>')
def click_emotion(userId=None, diaryId=None):
    API.click_emotion(userId, diaryId)
    return '', 202

# 다이어리 클릭 시 토픽 추가


@app.route('/click-diary/<userId>/<diaryId>')
def click_diary(userId=None, diaryId=None):
    API.click_diary(userId, diaryId)
    return '', 202
# recommand


@app.route('/recommand/<userId>')
def recommand(userId):
    result = API.recommand(userId)
    return jsonify({'diaries': result})

# 다이어리 생성시,수정 시  두 개 벡터 추가 및 토픽 추가 테스트


@app.route('/diary/create', methods=['POST'])
def vector():
    params = request.get_json()
    API.vector(params['diary_id'], params['title'], params['content'])
    return '', 202

# white list


# @app.before_request
# def limit_remote_addr():
#     if request.remote_addr != '127.0.1.1':
#         abort(403)  # Forbidden


if __name__ == "__main__":
    app.run(debug=True)
    # host 등을 직접 지정하고 싶다면
    # app.run(host="127.0.0.1", port="5000", debug=True)
