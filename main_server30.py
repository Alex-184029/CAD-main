from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import socket
from celery import Celery
from flask_bcrypt import Bcrypt
import requests

from alex.tools1 import parseTimeStr, getUUID, parseResult, write_dict_to_json, read_json_to_dict, readBase64, writeBase64

app = Flask(__name__)
CORS(app)
# MySQL所在主机名
HOSTNAME = "127.0.0.1"
# MySQL监听的端口号，默认3306
PORT = 3306
# 连接MySQL的用户名，自己设置
USERNAME = "root"
# 连接MySQL的密码，自己设置
PASSWORD = "123456"
# MySQL上创建的数据库名称
DATABASE = "cad"
# 通过修改以下代码来操作不同的SQL比写原生SQL简单很多 --》通过ORM可以实现从底层更改使用的SQL
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Celery异步任务框架 需要redis，尝试能否跑起来
app.config['broker_url'] = 'redis://localhost:6379/1'
app.config['result_backend'] = 'redis://localhost:6379/2'

celery = Celery(app.name, broker=app.config['broker_url'], backend=app.config['result_backend'])
celery.conf.update(app.config)

# 不存在则创建，可能没啥用
# with app.app_context():
#     db.create_all()

dwg_public = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public\dwgs3'
img_public = r'E:\School\Grad1\CAD\MyCAD2\CAD-main\dwg_file\public\dwgs3'

class User(db.Model):
    __tablename__ = 'user'
    user_id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(50))
    password = db.Column(db.String(50))
    role = db.Column(db.Enum('normal', 'super'))
    authority = db.Column(db.Enum('r', 'rw'))
    email = db.Column(db.String(50))
    phone_number = db.Column(db.String(50))

    def to_dict(self):
        return {'user_id': self.user_id, 'user_name': self.user_name, 'password': self.password,
                'role': self.role, 'authority': self.authority, 'email': self.email, 'phone_number': self.phone_number}


# class Task(db.Model):
#     __tablename__ = 'task'
#     task_id = db.Column(db.Integer, primary_key=True)
#     task_name = db.Column(db.String(50))
#     status = db.Column(db.Enum('processing', 'sucess', 'fail'))
#     create_time = db.Column(db.String(50))
#     process_time = db.Column(db.String(50))
#     user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
#     drawing_name = db.Column(db.String(50), db.ForeignKey('dwg_file.drawing_name'), nullable=False)
#     user = db.relationship('User', backref=db.backref('task', lazy=True))
#     dwg_file = db.relationship('Dwg_file', backref=db.backref('task', lazy=True))

#     def to_dict(self):
#         return {'task_id': self.task_id, 'task_name': self.task_name, 'status': self.status,
#                 'create_time': self.create_time, 'process_time': self.process_time, 'user_id': self.user_id,
#                 'drawing_name': self.drawing_name}

class Task(db.Model):    # 非外键约束Task
    __tablename__ = 'task2'
    task_id = db.Column(db.Integer, primary_key=True)
    task_name = db.Column(db.String(50))
    status = db.Column(db.Enum('processing', 'sucess', 'fail'))
    create_time = db.Column(db.String(50))
    process_time = db.Column(db.String(50))
    user_id = db.Column(db.Integer, nullable=False)
    drawing_name = db.Column(db.String(50), nullable=False)

    def to_dict(self):
        return {'task_id': self.task_id, 'task_name': self.task_name, 'status': self.status,
                'create_time': self.create_time, 'process_time': self.process_time, 'user_id': self.user_id,
                'drawing_name': self.drawing_name}


class Dwg_file(db.Model):
    __tablename__ = 'dwg_file'
    drawing_name = db.Column(db.String(50), primary_key=True)
    path = db.Column(db.String(50))
    result_id = db.Column(db.Integer, db.ForeignKey('result.result_id'), nullable=False)
    result = db.relationship('Result', backref=db.backref('dwg_file', lazy=True))

    def to_dict(self):
        return {'drawing_name': self.drawing_name, 'path': self.path, 'result_id': self.result_id}


class Temp_file(db.Model):
    __tablename__ = 'temp_file'
    file_name = db.Column(db.String(50), primary_key=True)
    path = db.Column(db.String(50))
    drawing_name = db.Column(db.String(50), db.ForeignKey('dwg_file.drawing_name'), nullable=False)
    dwg_file = db.relationship('Dwg_file', backref=db.backref('temp_file', lazy=True))

    # def __repr__(self):
    #     return '<Temp_File file_name=%r path=%r>' % (self.file_name, self.path)

    def to_dict(self):
        return {'file_name': self.file_name, 'path': self.path, 'drawing_name': self.drawing_name}


class Result(db.Model):
    __tablename__ = 'result'
    result_id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.Text)

    def to_dict(self):
        return {'result_id': self.result_id, 'data': self.data}

def send_msg(message):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8088))   # 连接本地socket端口

    # 发送字符串数据到服务器端
    # message = "I'm fine, thanks."
    client_socket.send(message.encode('utf-8'))

    data = client_socket.recv(1024).decode('utf-8')
    print('res:', data)
    client_socket.close()
    return data

def post_url_window(url, dwgpath):
    files = {'file': open(dwgpath, 'rb')}
    response = requests.post(url, verify=False, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        return None


# 用户管理接口
@app.route('/register', methods=['POST'])
def register():     # 注册
    # 假设你通过表单或其他方式接收了username和email
    user_id = request.form['user_id']
    user_name = request.form['user_name']
    password = request.form['password']
    role = request.form['role']
    authority = request.form['authority']
    email = request.form['email']
    phone_number = request.form['phone_number']
    print('user_name:', user_name)
    print('passowrd:', password)

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    print('hashed_password:', hashed_password)

    if User.query.filter_by(user_id=user_id).first():    # user_id用户的唯一性
        return jsonify({'message': 'register fail'}), 200
    # 创建新用户实例
    new_user = User(user_id=user_id, user_name=user_name, password=hashed_password, role=role, authority=authority,
                    email=email, phone_number=phone_number)

    # 添加到会话并提交
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'register successful'}), 200


@app.route('/show_user')
def show_user():  # put application's code here
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])


@app.route('/query_user', methods=['POST'])
def query_user():
    user_id = request.form['user_id']
    print(user_id)
    # print(type(user_id))
    target_user = User.query.get(user_id)

    return jsonify(target_user.to_dict())


@app.route('/login', methods=['POST'])
def login():
    user_name = request.form['user_name']
    password = request.form['password']
    # print('username:', user_name)
    # print('password:', password)
    # print(type(user_id))
    target_user = User.query.filter_by(user_name=user_name).first()
    # bcrypt.check_password_hash(user.password, password)
    # if target_user and password == target_user.to_dict()['password']:
    # print('database password:', target_user.to_dict()['password'])
    if target_user and bcrypt.check_password_hash(target_user.to_dict()['password'], password):
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid username or password'}), 200


@app.route('/query_user_by_name', methods=['POST'])
def query_user_by_name():
    user_name = request.form['name']
    print(user_name)
    # print(user_id)
    # print(type(user_id))
    target_user = User.query.filter(User.user_name == user_name).first()

    return jsonify(target_user.to_dict())


@app.route('/modify_info', methods=['POST'])
def modify_info():
    user_id = request.form['user_id']
    user_name = request.form['user_name']
    password = request.form['password']
    email = request.form['email']
    phone_number = request.form['phone_number']

    target_user = User.query.filter_by(user_id=user_id).first()
    if target_user:
        target_user.user_name = user_name
        target_user.password = password
        target_user.email = email
        target_user.phone_number = phone_number
        try:
            db.session.commit()
            return jsonify({'message': 'modify successful'}), 200
        except Exception as e:
            db.session.rollback()  # 如果出现异常，回滚事务
    return jsonify({'message': 'modify fail'}), 200

def add_temp_file(file_name, path, drawing_name):
    new_temp_file = Temp_file(file_name=file_name, path=path, drawing_name=drawing_name)
    db.session.add(new_temp_file)
    db.session.commit()

    return 'Temp_File added.'

# 任务管理
@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = main_worker.AsyncResult(task_id)    # 解析到这里时main_worker还没定义
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@celery.task(bind=True)
def main_worker(self, task_id):
    print('here is main worker, task_id:', task_id)

    # 任务处理函数
    # res = recongDoor(task_id)
    res = recongDoor2(task_id)
    print('recong door finish, ', res)


@app.route('/submit_task', methods=['POST'])
def submit_task():
    task_id = getUUID()
    task_name = request.form['task_name']
    status = request.form['status']
    create_time = request.form['create_time']
    process_time = request.form['process_time']
    user_id = request.form['user_id']
    drawing_name = request.form['drawing_name']

    if Task.query.filter_by(task_id=task_id).first():    # id已存在
        return jsonify({'message': 'submit task fail'}), 200
    # 创建新任务实例
    new_task = Task(task_id=task_id, task_name=task_name, status=status, create_time=create_time,
                    process_time=process_time, user_id=user_id, drawing_name=drawing_name)

    db.session.add(new_task)
    db.session.commit()

    task = main_worker.apply_async(args=[task_id])
    return jsonify({'message': 'submit task successful', 'Location': url_for('taskstatus', task_id=task.id)}), 200


@app.route('/query_task', methods=['POST'])
def query_task():
    task_id = request.form['task_id']
    print(task_id)
    # print(type(user_id))
    target_task = Task.query.get(task_id)

    return jsonify(target_task.to_dict())


@app.route('/query_task_list', methods=['POST'])
def query_task_list():
    user_id = request.form['user_id']
    task_list = Task.query.filter(Task.user_id == user_id).all()
    dict_list = [task.to_dict() for task in task_list]
    dict_list = sorted(dict_list, key=lambda x: parseTimeStr(x['create_time']), reverse=True)    # 时间排序
    return jsonify(dict_list)


@app.route('/query_task_list2', methods=['POST'])
def query_task_list2():
    user_id = request.form['user_id']
    page = int(request.form['page'])
    limit = int(request.form['limit'])
    task_list = Task.query.filter(Task.user_id == user_id).all()
    dict_list = [task.to_dict() for task in task_list]
    dict_list = sorted(dict_list, key=lambda x: parseTimeStr(x['create_time']), reverse=True)    # 时间排序

    total = len(dict_list)
    x = min((page - 1) * limit, total)
    y = min(page * limit, total)
    form = dict_list[x:y]
    return jsonify({'form': form, 'total': total})


@app.route('/delete_task', methods=['POST'])
def delete_task():
    task_id = request.form['task_id']
    task_to_delete = Task.query.get(task_id)
    if task_to_delete:
        # 删除
        db.session.delete(task_to_delete)
        # 提交事务
        db.session.commit()
        return jsonify({'message': 'delete succeed'}), 200
    else:
        # 不存在
        return jsonify({'message': 'delete fail'}), 200

@app.route('/dwg', methods=['POST'])
def get_dwg_file():     # 下载dwg文件到本地
    file = request.files['file']

    if file:
        filename = file.filename
        # 文件保存本地
        file.save(os.path.join(dwg_public, filename))

        # add_dwg_file(filename, os.path.join('./dwg_file/', filename))

        json_str = {'result': 'success', 'filename': filename}

    else:
        json_str = {'result': 'fail'}

    return jsonify(json_str)

@app.route('/get-image_door', methods=['POST'])
def get_image_door():
    drawing_name = os.path.splitext(request.form['drawing_name'])[0]
    try:
        imgdoor = os.path.join(img_public, drawing_name + '_Origin.jpg')
        print('imgdoor:', imgdoor)
        with open(imgdoor, 'rb') as img_file:
            response = make_response(img_file.read())
            response.headers.set('Content-Type', 'image_door/png')
            return response
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/get-item_list0', methods=['POST'])
def get_item_list0():
    dwgname = os.path.splitext(request.form['drawing_name'])[0]
    try:
        logfile = os.path.join(dwg_public, (dwgname + '_ArcDoor.txt'))
        res = parseResult(logfile)
        if res is None:
            print('res is None')
            return jsonify({'error': 'Get item list none.'}), 404
        # print('box:', res['box'])
        # print('total:', res['total'])
        # print('rects:', res['rects'])
        return jsonify(res), 200
    except FileNotFoundError:
        return jsonify({'error': 'Get item list fail.'}), 404

@app.route('/get-item_list', methods=['POST'])
def get_item_list():
    dwgname = os.path.splitext(request.form['drawing_name'])[0]
    try:
        logfile = os.path.join(dwg_public, (dwgname + '_ArcDoor.json'))
        res = read_json_to_dict(logfile)
        # print('get item list res:', res)
        if res is None:
            print('res is None')
            return jsonify({'error': 'Get item list none.'}), 404
        # print('box:', res['box'])
        # print('total:', res['total'])
        # print('rects:', res['rects'])
        return jsonify(res), 200
    except FileNotFoundError:
        return jsonify({'error': 'Get item list fail.'}), 404

@app.route('/recong_door', methods=['POST'])
def recong_door():
    dwgname = request.form['dwgname']
    print('dwgname:', dwgname)
    dwgpath = os.path.join(dwg_public, dwgname)
    print('dwgpath:', dwgpath)
    # 门算法解析，并且返回图片名（图片名可以与图纸文件名保持一致）

    return jsonify({'message': 'Recong door successful'}), 200

@app.route('/recong_door2', methods=['POST'])
def recong_door2():
    dwgname = request.form['dwgname']
    print('dwgname:', dwgname)
    imgname = 'tmp.jpg'
    imgpath = os.path.join(img_public, imgname)
    print('imgpath:', imgpath)
    base64img = readBase64(imgpath)

    return jsonify({'img_base64': base64img}), 200

def recongDoor(task_id):
    print('Here is recongDoor.')
    with app.app_context():
        target_task = Task.query.filter_by(task_id=task_id).first()
        dwgname = target_task.drawing_name
        print('图纸%s开始统计' % dwgname)
        res = send_msg('ArcDoor ' + dwgname)
        if res == 'Succeed':
            print('图纸%s统计完毕' % dwgname)
            target_task.status = 'sucess'     # 更新任务状态，注意是sucess
        elif res == 'Fail':
            print('图纸%s统计完毕' % dwgname)
            target_task.status = 'fail'
        else:
            console += '\n出现未知Socket信息。\n'
            target_task.status = 'fail'

        try:
            db.session.commit()
            return 'success'
        except Exception as e:
            print('error:', e)
            db.session.rollback()  # 如果出现异常，回滚事务
            return 'fail'

def recongDoor2(task_id):
    print('Here is recongDoor2.')
    # url = 'http://127.0.0.1:5005/parse_door'        # 本地
    url = 'http://192.168.131.128:5005/parse_door'  # 虚拟机
    with app.app_context():
        target_task = Task.query.filter_by(task_id=task_id).first()
        dwgname = target_task.drawing_name
        print('图纸 %s 开始统计' % dwgname)

        # 提交请求进行解析
        try:
            res = post_url_window(url, os.path.join(dwg_public, dwgname))
            if res is None or res['status'] != 'success':
                print('post_url_window res:', res)
                return None
            print('res satatus', res['status'])
            res_parse = res['res']
            tmpname = os.path.splitext(dwgname)[0]
            jsonpath = os.path.join(dwg_public, tmpname + '_ArcDoor.json')
            print('jsonpath:', jsonpath)
            write_dict_to_json(res_parse, jsonpath)
            imgpath = os.path.join(img_public, tmpname + '_Origin.jpg')
            writeBase64(res['img'], imgpath)
            target_task.status = 'sucess'
        except Exception as e:
            print('error in recongDoor2:', e)
            target_task.status = 'fail'

        # 事务提交
        try:
            db.session.commit()
            return 'success'
        except Exception as e:
            print('error:', e)
            db.session.rollback()  # 如果出现异常，回滚事务
            return 'fail'

def closeSocket():
    send_msg('Terminate')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
