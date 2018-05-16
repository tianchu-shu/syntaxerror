class Connect():
    def __init__(self):
        self.conn = None
        self.cur = None
    

    def connect_to():
        
        params = {
            'dbname': 'jocodssg',
            'user': 'jocodssg_students',
            'host': 'postgres.dssg.io',
            'password': 'aibaecighoobeeba',
            'port': 5432
        }

        try:
            conn = psycopg2.connect(**params)
            print("connected successfully.")
        except:
            print("failed to connect.")

        self.conn = self.conn.cursor()


    def close_connect(self):
        try:
            self.cur.close()
            self.conn.close()
            print("closed connection.")
        except:
            print("failed to close connection.")



