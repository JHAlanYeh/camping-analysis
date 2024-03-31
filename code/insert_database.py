import os
import json
import pymysql

db_settings = {
    "host" : "127.0.0.1",
    "port" : 3306,
    "user" : "admin" ,
    "password" : "z]J_oLAr)T8/M75a" ,
    "db" : "camping",
    "charset" : "utf8",
}


def asiayo_insert_database():
    mas_sql="""INSERT INTO camping_areas (name, address, longitude, latitude, description, website, category) VALUES (%s, %s, %s, %s, %s, %s, %s);"""
    dtl_sql="""INSERT INTO camping_comments (camping_id, comment_date, rating, content) VALUES (%s, %s, %s, %s);"""
    
    dir_path = os.path.join(os.getcwd(), "data\\asiayo_comments")
    file_list = os.listdir(dir_path)
    
    for file in file_list:
        print("=======================")
        print(file)
        file_path = os.path.join(dir_path, file)
        f = open(file_path, encoding="utf-8-sig")
        data = json.load(f)

        conn = pymysql.connect(**db_settings)
        try:
            with conn.cursor() as cursor:
                name = data["name"]
                if "｜" in data["name"]:
                    name = data["name"].split("｜")[0].strip()
                elif "|" in data["name"]:
                    name = data["name"].split("|")[0].strip()
                mas_data = (name, data["address"], data["longitude"], data["latitude"], data["description"], "Asiayo", 0)
                cursor.execute(mas_sql, mas_data)
                camping_id = cursor.lastrowid

                for c in data["comments"]:
                    dtl_data = (camping_id, c["publishedDate"], c["rating"], c["content"])
                    cursor.execute(dtl_sql, dtl_data)
            conn.commit()
            print("finish insert to db")        
        except Exception as e:
            print(e.args)
            conn.rollback()
        print("=======================")


def easycamp_insert_database():
    mas_sql="""INSERT INTO camping_areas (name, address, longitude, latitude, description, website, category) VALUES (%s, %s, %s, %s, %s, %s, %s);"""
    dtl_sql="""INSERT INTO camping_comments (camping_id, comment_date, rating, content) VALUES (%s, %s, %s, %s);"""
    
    dir_path = os.path.join(os.getcwd(), "data\\easycamp_comments")
    file_list = os.listdir(dir_path)
    
    for file in file_list:
        file_path = os.path.join(dir_path, file)
        f = open(file_path, encoding="utf-8-sig")
        data = json.load(f)
        name = data["name"].split(" ")[-1].strip()
        conn = pymysql.connect(**db_settings)
        try:
            with conn.cursor() as cursor:
                name = data["name"]
                if "｜" in data["name"]:
                    name = data["name"].split("｜")[0].strip()
                elif "|" in data["name"]:
                    name = data["name"].split("|")[0].strip()
                
                if "latitude" not in data:
                    data["latitude"] = None
                if "longitude" not in data:
                    data["longitude"] = None
                mas_data = (name, data["address"], data["longitude"], data["latitude"], data["description"], "EasyCamp", 0)
                cursor.execute(mas_sql, mas_data)
                camping_id = cursor.lastrowid

                for c in data["comments"]:
                    dtl_data = (camping_id, c["publishedDate"], c["rating"], c["content"])
                    cursor.execute(dtl_sql, dtl_data)
            conn.commit()
            print("finish insert to db")        
        except Exception as e:
            print(e.args)
            # conn.rollback()

if __name__ == "__main__":
    asiayo_insert_database()
    easycamp_insert_database()