from pymongo import MongoClient

database_loc = 'mongodb://localhost:27017'
database_name = 'cptc2018'
collection_name = 'team_data'
api_username = 'api'
api_password = 'mongo_api'
use_auth=True

class DatabaseInterface :

	_client = None
	_db = None

	def __init__(self, url=None, db=None, collection=None, username=api_username, password=api_password):
		if url is None : self.init_db(database_loc, database_name, collection_name, api_username, api_password)
		else :
			self.init_db(url, db, collection, username, password)

	def main(self):

		print('Connecting to ' +  database_name + ' at ' + database_loc)
		client = self.get_dbclient(database_loc)
		db = self.get_database(database_name)

		print('Accessing collection ' + collection_name + ' and testing access.')
		coll = db[collection_name]
		test_alert = coll.find_one()
		if test_alert is None : print('Error: Nothing returned from query.')
		else : print('Success!')

		return

	def get_dbclient(self, url, username, password) :

		if use_auth :
			self._client = MongoClient(url, username=username, password=password, authSource='admin')
		else :	
			self._client = MongoClient(url)
			
		return self._client

	def get_database(self) :
		if self._client is None : raise Exception('Database not initialized or connected.')
		return self._db
		
	def get_collection(self, collection) :
		if self._client is None : raise Exception('Database not initialized or connected.')
		return self._db[collection]

	def set_collection(self, collection):

		if self._client is None: raise Exception('Database not initialized or connected.')
		self._db = self._db[collection]

	def init_db(self, url, db, collection, username, password) :

		client = self.get_dbclient(url, username, password)
		self._db = client[db][collection]
		
		return self._db

	def push_document(self, doc):
		if self._db is None : raise Exception('Database not initialized or connected.')
		return self._db.insert_one(doc)

	def push_documents(self, docs):
		if self._db is None : raise Exception('Database not initialized or connected.')
		return self._db.insert_many(docs)


if __name__ == '__main__':
	DatabaseInterface(None,None,None).main()