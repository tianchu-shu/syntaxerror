class Connect():
    def __init__(self):
        self.conn = None
        self.cur = None
        
        
        self.connect_to()
    

    def connect_to(self):
        
        params = {
            'dbname': 'jocodssg',
            'user': 'jocodssg_students',
            'host': 'postgres.dssg.io',
            'password': 'aibaecighoobeeba',
            'port': 5432
        }

        try:
            self.conn = psycopg2.connect(**params)
            print("connected successfully.")
        except:
            print("failed to connect.")

        self.conn = self.conn.cursor()
        print("Open Connection")


    def close_connect(self):
        if self.conn.closed == 0:
            try:
                self.cur.close()
                self.conn.close()
                print("Closed connection.")
            except:
                print("Failed to close connection.")



