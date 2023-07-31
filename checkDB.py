import sqlite3
connection = sqlite3.connect("db.sqlite3")
cursor = connection.execute('select * from images_isp')
print(cursor.fetchall())
