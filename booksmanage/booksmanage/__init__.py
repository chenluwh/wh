import pymysql
pymysql.install_as_MySQLdb()
'''Django连接MySQL时默认使用MySQLdb驱动，
但MySQLdb不支持Python3，
因此这里将MySQL驱动设置为pymysql
'''

# 创建与数据库连接对象
db = pymysql.connect('localhost', 'root', password="clwh243658", db="booksmanage", port=3306, charset="utf8")

# 利用db方法创建游标对象
cursor = db.cursor()

# 利用游标对象execute()方法执行SQL语句，()里填写正确的SQL语句
cursor.execute("SELECT VERSION()")

# 使用fetchone()获取一条数据库
data = cursor.fetchone()

print("Database version : %s" % data)

# 关闭数据库连接
db.close()
