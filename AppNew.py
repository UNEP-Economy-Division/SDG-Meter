from flask import Flask, render_template, request, jsonify

# from flask_socketio import SocketIO, join_room
# import eventlet
from flask import jsonify

# from unep import read_file
from datetime import datetime
from io import StringIO
from Bert import get_SDG
import operator
# eventlet.monkey_patch()
import json

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
from flask_cors import CORS
CORS(app)
app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)
# # 用来启动一个协程进行 unep 当中的分析操作
# def socket_analyse(f):
    # def wrapper(*args,**kwargs):
        # room = kwargs.pop('room')
        # res = f(*args,**kwargs)
        # socketio.emit('data',res,broadcast=False,room=room)
    # return wrapper

# read_file = socket_analyse(read_file)

# @socketio.on('upload')
# def analyse(file):
    # room = str(datetime.now().timestamp())
    # join_room(room=room)
    # file = StringIO(file.decode())
    # socketio.emit('upload','Successfully uploaded the file, please wait for analyse!')
    # socketio.start_background_task(read_file(file,room=room))

@app.route('/',methods=['GET'])
def indexNew():
    return "live"

@app.route('/sdgOP',methods=['POST'])
def index():
    content=request.json
    data = get_SDG(content["text"])
    data_ = json.loads(data)
    data_lists = data_['data']
    cars_dict = {data_list[0]: data_list[1] for data_list in data_lists}
    new_data = {k: v for k, v in cars_dict.items() if v >= 0.015}
    new_data = {k: int(v * 100) for k, v in new_data.items()}
    new_data = sorted(new_data.items(), key=operator.itemgetter(1), reverse=True)
    sortdict = dict(new_data)
    print(sortdict)
    #
    response =jsonify(sortdict)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    #
    # return jsonify(sortdict)


@app.route('/default', methods=['POST', 'GET'])
def default():
    data = {
        'SDG12': 0.92919921875,
        'SDG6': 0.034088134765625,
        'SDG7': 0.0250396728515625,
        'SDG11': 0.021697998046875,
        'SDG8': 0.017181396484375,
        'SDG4': 0.0144500732421875,
        'SDG2': 0.0135345458984375,
        'SDG14': 0.01322174072265625,
        'SDG3': 0.012481689453125,
        'SDG16': 0.011505126953125,
        'SDG13': 0.01094818115234375,
        'SDG9': 0.0095977783203125,
        'SDG15': 0.008575439453125,
        'SDG10': 0.00841522216796875,
        'SDG5': 0.00714874267578125
    }
    new_data = {k: v for k, v in data.items() if v >= 0.015}
    new_data = {k: int(v*100) for k, v in new_data.items()}

    new_data = sorted(new_data.items(), key=operator.itemgetter(1), reverse=True)
    sortdict = dict(new_data)
    print(sortdict)
    return jsonify(sortdict)



if __name__ == '__main__':
    app.run(debug=False)
