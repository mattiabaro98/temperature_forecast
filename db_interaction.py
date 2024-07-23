import psycopg2


class db_interaction:

    def __init__(self, dbname, user, password, host, port):

        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def create_table(self):
        try:
            connection = psycopg2.connect(
                dbname=self.dbname, user=self.user, password=self.password, host=self.host  # , port=self.port
            )
            cursor = connection.cursor()
            create_table_query = """
            CREATE TABLE IF NOT EXISTS th_table (
                timestamp BIGINT,
                device_id VARCHAR(50),
                temperature FLOAT,
                humidity FLOAT
            )
            """
            cursor.execute(create_table_query)
            connection.commit()
            cursor.close()
            connection.close()
        except Exception as error:
            print(f"Error creating table: {error}")

    def drop_table(self):
        try:
            connection = psycopg2.connect(
                dbname=self.dbname, user=self.user, password=self.password, host=self.host, port=self.port
            )
            cursor = connection.cursor()

            drop_table_query = "DROP TABLE IF EXISTS th_table;"
            cursor.execute(drop_table_query)
            connection.commit()
            cursor.close()
            connection.close()
            print("Table deleted successfully.")
        except Exception as error:
            print(f"Error deleting table: {error}")

    def write_record_data(self, records, event=None):
        try:
            connection = psycopg2.connect(
                dbname=self.dbname, user=self.user, password=self.password, host=self.host, port=self.port
            )
            cursor = connection.cursor()
            insert_query = """
            INSERT INTO th_table (timestamp, device_id, temperature, humidity) VALUES (%s, %s, %s, %s)
            """
            cursor.executemany(insert_query, records)
            connection.commit()
            cursor.close()
            connection.close()
        except Exception as error:
            print(f"Error writing data: {error}")
            if event:
                event.set()

    def read_record_data(self, start_time, end_time):
        try:
            connection = psycopg2.connect(
                dbname=self.dbname, user=self.user, password=self.password, host=self.host, port=self.port
            )
            cursor = connection.cursor()
            select_query = """
            SELECT timestamp, device_id, temperature, humidity FROM th_table WHERE timestamp BETWEEN %s AND %s
            """
            cursor.execute(select_query, (start_time, end_time))
            records = cursor.fetchall()
            cursor.close()
            connection.close()
            return records
        except Exception as error:
            print(f"Error reading data: {error}")
            return []
