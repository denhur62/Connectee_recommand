import MySQLdb
from .config import lnfo


def db_execute(query, arg=None, many=False):
    conn = MySQLdb.connect(
        user=lnfo['user'],
        passwd=lnfo['passwd'],
        host=lnfo['host'],
        db=lnfo['db'],
        port=lnfo['port'],
        charset=lnfo['charset']
    )
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    res = ''
    if arg == None:
        try:
            cur.execute(query)
            res = cur.fetchall()
            conn.commit()
        except Exception as e:
            print(e)
            print("arg none")
        finally:
            cur.close()
            conn.close()
    else:
        if many == True:
            try:
                cur.executemany(query, arg)
                res = cur.fetchall()
                conn.commit()
            except Exception as e:
                print(e)
                print("many")
            finally:
                cur.close()
                conn.close()
        else:
            try:
                cur.execute(query, arg)
                res = cur.fetchall()
                conn.commit()
            except Exception as e:
                print(e)
                print("alone")
            finally:
                cur.close()
                conn.close()

    return res
