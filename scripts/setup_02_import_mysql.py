# -*- coding: utf-8 -*-
"""建 jpkb 表并导入高频问答 CSV 到 MySQL"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mysql_qa.db.mysql_client import MySQLClient

CSV_PATH = os.path.join(ROOT, "mysql_qa", "data", "JP学科知识问答.csv")

client = MySQLClient()
client.create_table()
client.cursor.execute("SELECT COUNT(*) FROM jpkb")
before = client.cursor.fetchone()[0]
print(f"导入前 jpkb 行数: {before}")
if before == 0:
    client.insert_data(CSV_PATH)
client.cursor.execute("SELECT COUNT(*) FROM jpkb")
after = client.cursor.fetchone()[0]
print(f"导入后 jpkb 行数: {after}")
client.close()
