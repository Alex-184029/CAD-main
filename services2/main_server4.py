from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import base64
import socket
from celery import Celery
from flask_bcrypt import Bcrypt
import shutil

from tools.tools1 import parseTimeStr, getUUID, parseResult
from tools.parse_cad import parse_door, parse_wall, parse_window, parse_area, dwg_public
from tools.tools_result import parse_result, parse_bill

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
    task_type = db.Column(db.String(50), nullable=False)

    def to_dict(self):
        return {'task_id': self.task_id, 'task_name': self.task_name, 'task_type': self.task_type, 'status': self.status,
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

def readBase64(imgpath):
    if not os.path.exists(imgpath):
        return None
    with open(imgpath, 'rb') as img_file:
        img_data = img_file.read()
        base64_img = base64.b64encode(img_data)
        base64_img_str = base64_img.decode('utf-8')
    return base64_img_str

def getBase64Image(task_id, drawing_name, task_type):
    work_dir = os.path.join(dwg_public, task_id)
    dwg = os.path.splitext(drawing_name)[0]
    blank_path = os.path.join(dwg_public, 'blank.jpg')
    if task_type == 'LegendMatch':     # 还未加入系统的图例匹配功能
        return readBase64(blank_path)  # 先返回空吧
    else:
        img_path = os.path.join(work_dir, dwg + '_PlaneLayout.jpg')
        imgstr = readBase64(img_path)
        return readBase64(blank_path) if imgstr is None else imgstr

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
def main_worker(self, task_id, task_type):
    print('here is main worker, task_id: %s, task_type: %s' % (task_id, task_type))

    task_type_list = []
    if ',' in task_type:
        task_type_list = task_type.split(',')
    else:
        task_type_list.append(task_type)
    # print('task_type_list:', task_type_list)

    if 'Area' in task_type_list:     # 区域包含其它功能
        res = recongArea(task_id)
        print('recong window finish, ', res)
        return

    for task_type in task_type_list:
        if task_type == 'Door':
            res = recongDoor(task_id)
            print('recong door finish, ', res)
        elif task_type == 'Window':
            res = recongWindow(task_id)
            print('recong window finish, ', res)
        elif task_type == 'Wall':
            res = recongWall(task_id)
            print('recong window finish, ', res)
        else:
            print('Task type not supported.')


@app.route('/submit_task', methods=['POST'])
def submit_task():
    task_id = getUUID()
    task_name = request.form['task_name']
    status = request.form['status']
    create_time = request.form['create_time']
    process_time = request.form['process_time']
    user_id = request.form['user_id']
    drawing_name = request.form['drawing_name']

    task_type = request.form['task_type']

    if Task.query.filter_by(task_id=task_id).first():    # id已存在
        return jsonify({'message': 'submit task fail'}), 200
    # 创建新任务实例
    new_task = Task(task_id=task_id, task_name=task_name, status=status, create_time=create_time,
                    process_time=process_time, user_id=user_id, drawing_name=drawing_name, task_type=task_type)

    db.session.add(new_task)
    db.session.commit()

    # 构建任务工作目录
    task_dir = os.path.join(dwg_public, task_id)
    os.makedirs(task_dir, exist_ok=True)
    dwg_path = os.path.join(dwg_public, drawing_name)
    if not os.path.exists(dwg_path):
        return jsonify({'message': f'dwg_path {dwg_path} not exist'}), 404
    shutil.move(dwg_path, task_dir)

    task = main_worker.apply_async(args=[task_id, task_type])
    return jsonify({'message': 'submit task successful', 'Location': url_for('taskstatus', task_id=task.id)}), 200


@app.route('/query_task', methods=['POST'])
def query_task():
    task_id = request.form['task_id']
    print(task_id)
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


@app.route('/query_task_list3', methods=['POST'])
def query_task_list3():
    user_id = request.form['user_id']
    page = int(request.form['page'])
    limit = int(request.form['limit'])
    select_title = request.form['title']
    select_type = request.form['type']
    select_sort = request.form['sort']
    print('select info:', select_title, select_type, select_sort)
    
    dict_list = []
    if select_type == 'All':
        task_list = Task.query.filter(Task.user_id == user_id).filter(Task.task_name.like(f'%{select_title}%')).all()
        dict_list = [task.to_dict() for task in task_list]
    else:
        task_list = Task.query.filter(Task.user_id == user_id).filter(Task.task_name.like(f'%{select_title}%')).filter(Task.task_type == select_type).all()
        dict_list = [task.to_dict() for task in task_list]

    # print('dict list length:', len(dict_list))
    if select_sort == '-time':
        dict_list = sorted(dict_list, key=lambda x: parseTimeStr(x['create_time']), reverse=True)    # 时间倒序
    elif select_sort == '+time':
        dict_list = sorted(dict_list, key=lambda x: parseTimeStr(x['create_time']), reverse=False)    # 时间倒序
    else:
        print('Error: The format of select_sort is wrong.', select_sort)

    total = len(dict_list)
    x = min((page - 1) * limit, total)
    y = min(page * limit, total)
    form = dict_list[x:y]

    # 加入base64图片
    for item in form:
        item['img_base64'] = getBase64Image(item['task_id'], item['drawing_name'], item['task_type'])

    return jsonify({'form': form, 'total': total})


@app.route('/delete_task', methods=['POST'])      # 老师觉得没必要删除任务，未使用
def delete_task():
    task_id = request.form['task_id']
    task_to_delete = Task.query.get(task_id)
    if task_to_delete:
        # 删除
        db.session.delete(task_to_delete)
        # 提交事务
        db.session.commit()
        work_dir = os.path.join(dwg_public, task_id)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
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

@app.route('/get-image-bill', methods=['POST'])
def get_image_bill():
    drawing_name = request.form['drawing_name']
    task_id = request.form['task_id']
    try:
        imgpath = os.path.join(dwg_public, task_id, drawing_name[:-4] + '_PlaneLayout.jpg')
        with open(imgpath, 'rb') as img_file:
            response = make_response(img_file.read())
            response.headers.set('Content-Type', 'image_PlaneLayout/jpg')
            return response
    except FileNotFoundError:
        return jsonify({'error': 'File not found.'}), 404

@app.route('/get-image', methods=['POST'])
def get_image():
    drawing_name = request.form['drawing_name']
    task_type = request.form['task_type']
    # 前端需要把task_id传过来
    task_id = request.form['task_id']
    plane_layout_list = ['Window', 'Door', 'Wall', 'Area']
    try:
        # if any(item in task_type for item in plane_layout_list):   # 这里的type为单一类型
        if task_type in plane_layout_list:
            imgpath = os.path.join(dwg_public, task_id, drawing_name[:-4] + '_PlaneLayout.jpg')
            with open(imgpath, 'rb') as img_file:
                response = make_response(img_file.read())
                response.headers.set('Content-Type', 'image_PlaneLayout/jpg')
                return response
        # 后续要加非平面布置图的视口图像
        else:
            return jsonify({'error': 'Task type error.'}), 404
    except FileNotFoundError:
        return jsonify({'error': 'File not found.'}), 404

@app.route('/get-item_list', methods=['POST'])
def get_item_list():
    dwgname = os.path.splitext(request.form['drawing_name'])[0]
    task_type = request.form['task_type']
    task_id = request.form['task_id']
    work_dir = os.path.join(dwg_public, task_id)
    try:
        logfile = os.path.join(work_dir, dwgname + '_Res' + task_type + '.json')
        res = parse_result(logfile, task_type)    # 传json中原始数据，前端决定如何显示
        if res is None:
            print('res is None')
            return jsonify({'error': 'Get item list none.'}), 404
        return jsonify(res), 200
    except:
        return jsonify({'error': 'Get item list fail.'}), 404

@app.route('/get-bill', methods=['POST'])
def get_bill():
    # dwgname = os.path.splitext(request.form['drawing_name'])[0]
    task_id = request.form['task_id']
    work_dir = os.path.join(dwg_public, task_id)
    try:
        logfile = os.path.join(work_dir, 'bill.json')
        res = parse_bill(logfile)
        if res is None:
            print('res is None')
            return jsonify({'error': 'Get bill none.'}), 404
        return jsonify(res), 200
    except:
        return jsonify({'error': 'Get bill fail.'}), 404

def recongDoor(task_id):
    print('Here is recongDoor.')
    with app.app_context():
        target_task = Task.query.filter_by(task_id=task_id).first()
        dwgname = target_task.drawing_name
        print('图纸%s开始解析' % dwgname)
        res = parse_door(task_id, dwgname)
        if res == 'Succeed':
            print('图纸%s解析成功。' % dwgname)
            target_task.status = 'sucess'
        else:
            print('图纸%s解析失败。' % dwgname)
            target_task.status = 'fail'
        try:
            db.session.commit()
            return 'success'
        except Exception as e:
            print('error:', e)
            db.session.rollback()  # 如果出现异常，回滚事务
            return 'fail'

def recongWindow(task_id):
    print('Here is recongWindow.')
    with app.app_context():
        target_task = Task.query.filter_by(task_id=task_id).first()
        dwgname = target_task.drawing_name
        print('图纸%s开始解析' % dwgname)
        res = parse_window(task_id, dwgname)
        if res == 'Succeed':
            print('图纸%s解析成功。' % dwgname)
            target_task.status = 'sucess'
        else:
            print('图纸%s解析失败。' % dwgname)
            target_task.status = 'fail'
        try:
            db.session.commit()
            return 'success'
        except Exception as e:
            print('error:', e)
            db.session.rollback()  # 如果出现异常，回滚事务
            return 'fail'

def recongWall(task_id):
    print('Here is recongWall.')
    with app.app_context():
        target_task = Task.query.filter_by(task_id=task_id).first()
        dwgname = target_task.drawing_name
        print('图纸%s开始解析' % dwgname)
        res = parse_wall(task_id, dwgname)
        if res == 'Succeed':
            print('图纸%s解析成功。' % dwgname)
            target_task.status = 'sucess'
        else:
            print('图纸%s解析失败。' % dwgname)
            target_task.status = 'fail'
        try:
            db.session.commit()
            return 'success'
        except Exception as e:
            print('error:', e)
            db.session.rollback()  # 如果出现异常，回滚事务
            return 'fail'

def recongArea(task_id):
    print('Here is recongArea.')
    with app.app_context():
        target_task = Task.query.filter_by(task_id=task_id).first()
        dwgname = target_task.drawing_name
        print('图纸%s开始解析' % dwgname)
        res = parse_area(task_id, dwgname)
        if res == 'Succeed':
            print('图纸%s解析成功。' % dwgname)
            target_task.status = 'sucess'
        else:
            print('图纸%s解析失败。' % dwgname)
            target_task.status = 'fail'
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
    app.run()
