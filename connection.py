import psycopg2
import numpy as np
import pandas as pd
import json
import logging


class Connect():
    def __init__(self):
        self.conn = None
        self.cur = None
        
        self.connect_to()
    

    def connect_to(self):
        
        params = {'dbname': 'jocodssg',
            'user': 'jocodssg_students',
            'host': 'postgres.dssg.io',
            'password': 'aibaecighoobeeba',
            'port': 5432}

        try:
            self.conn = psycopg2.connect(**params)
            print("Connected successfully.")
        except Exception as e:
            print("Failed to connect.")
            print(e)

        self.conn = self.conn.cursor()
        print("Open Connection")


    def close_connect(self):
        if self.conn.closed == 0:
            try:
                self.cur.close()
                self.conn.close()
                print("Closed connection.")
            except Exception as e:
                print("Failed to close connection.")
                print(e)
		else:
			print("No Connection")

	def print_df(self, type, input_string):
        if self.conn.closed == 0:
			if type = 'table':
				df = pd.read_sql_table(input_string, self.conn)
			elif type = 'query':
				df = pd.read_sql_query(input_string, self.conn)
			else:
				df = None
				print("Invalid 'Type'")
			return df
		else:
			print("No Connection")
