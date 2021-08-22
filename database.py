import MySQLdb
from config import lnfo


def db_execute(query, arg=None):
    conn = MySQLdb.connect(
        user=lnfo['user'],
        passwd=lnfo['passwd'],
        host=lnfo['host'],
        db=lnfo['db'],
        port=lnfo['port'],
        charset=lnfo['charset']
    )
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    if arg == None:
        try:
            cur.execute(query)
            res = cur.fetchall()
            conn.commit()
        except Exception as e:
            print(e)
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
        finally:
            cur.close()
            conn.close()
    return res
