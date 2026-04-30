import psycopg2

db_host = 'flight-price-3.c3yku68surp7.ap-south-2.rds.amazonaws.com'
db_name = 'flight_db'
db_user = 'gowtham'
db_password = 'naidu2733'

connection = psycopg2.connect(
    host=db_host,
    dbname=db_name,
    user=db_user,
    password=db_password
)
print("Database connected successfully")
cursor = connection.cursor()
cursor.execute('select version()')
db_version = cursor.fetchone()
print('PostgreSQL database version:', db_version)

cursor.close()