from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

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

def doTest():
    task_id = 76257
    with app.app_context():
        target_task = Task.query.filter_by(task_id=task_id).first()
        print('target_task0:', target_task.to_dict())

        target_task.status = 'sucess'
        print('target_task1:', target_task.to_dict())

        db.session.commit()

        target_task2 = Task.query.filter_by(task_id=task_id).first()
        print('target_task2:', target_task2.to_dict())
    print('-- finish --')


if __name__ == '__main__':
    doTest()