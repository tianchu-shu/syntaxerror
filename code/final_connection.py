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
	

	def load_params(self, filepath = 'params.json'):
		with open(filepath, 'r') as file:
			params = json.load(file)
		
		return params
	
	def connect_to(self):
		params = self.load_params()

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

	def run_query(self, query):
		try:
			self.cur.executre(query)
		except:
			print("Query Execution Failed, Rolling Back")
			self.conn.rollback()
			
	def return_df(self, type, input_string):
		if self.conn.closed == 0:
			try:
				if type == 'table':
					df = pd.read_sql_table(input_string, self.conn)
				elif type == 'query':
					df = pd.read_sql_query(input_string, self.conn)
				else:
					df = None
					print("Invalid 'Type'")
				return df
			except:
				print("Query Execution Failed, Rolling Back")
				self.conn.rollback()
		else:
			print("No Connection")
