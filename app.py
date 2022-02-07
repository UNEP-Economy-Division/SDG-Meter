from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room
import eventlet
from unep import read_file
from datetime import datetime
from io import StringIO

eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
# 用来启动一个协程进行 unep 当中的分析操作
def socket_analyse(f):
    def wrapper(*args,**kwargs):
        room = kwargs.pop('room')
        res = f(*args,**kwargs)
        socketio.emit('data',res,broadcast=False,room=room)
    return wrapper

read_file = socket_analyse(read_file)

@socketio.on('upload')
def analyse(file):
    room = str(datetime.now().timestamp())
    join_room(room=room)
    file = StringIO(file.decode())
    socketio.emit('upload','Successfully uploaded the file, please wait for analyse!')
    socketio.start_background_task(read_file(file,room=room))

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', debug=True)
