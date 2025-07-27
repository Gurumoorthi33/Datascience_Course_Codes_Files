import sqlite3

# connection = sqlite3.Connection("lmes.db")
# query = """ create table course(c_name text)"""
# connection.execute(query)
# connection.commit()
# connection.close()

connection = sqlite3.Connection("lmes.db")
query = """ insert into course values ('ds')"""
connection.execute(query)
connection.commit()
connection.close()

# ODBC ===> Open DataBase Connectivity

""" steps to connect with sqlite3 with PowerBI
1. Ensure that you have proper db file for sqlite3
2. Download odbc for sqlite3
   2.1 ODBC is the OpenSource Database Connectivity needed to connect our db with external tools
   2.2 version 32 or 64 bit
   2.3 Then finish the setup
3. Open the PowerBI
  3.1 choose datasources-----> Others ----> odbc
  3.2 upon the click of ODBC you shouid be seeing the file which you added recently """