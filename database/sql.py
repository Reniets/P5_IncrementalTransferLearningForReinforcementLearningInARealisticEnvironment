import mysql.connector
from database.dbSettings import DbSettings


class Sql:
    def __init__(self):
        self.db = mysql.connector.connect(
            host     = DbSettings.HOST.value,
            user     = DbSettings.USER.value,
            passwd   = DbSettings.PASS.value,
            database = DbSettings.DB.value,
        )

    def SELECT_allSessions(self):
        cursor = self._cursor()

        query = "SELECT * FROM sessions WHERE 1"
        cursor.execute(query)

        res = cursor.fetchall()
        cursor.close()

        return res

    def SELECT_trainingData(self, sessionId, evaluation):
        cursor = self._cursor()

        query = "SELECT `instance`, `number`, `reward` FROM `episodes` WHERE `session`=%s AND `evaluation`=%s AND `video_blob` IS NULL"
        cursor.execute(query, (sessionId, evaluation))
        res = cursor.fetchall()
        cursor.close()

        return res

    def INSERT_newSession(self, name):
        cursor = self._cursor()

        query = "INSERT INTO `sessions`(`name`) VALUES (%s)"
        cursor.execute(query, (name,))

        self.db.commit()

        return cursor.lastrowid

    def INSERT_newEpisode(self, session_id, instance, episode_nr, reward, evaluation=False, video_blob=None, model_blob=None):
        cursor = self._cursor()

        query = """
                    INSERT INTO `episodes`(`session`, `number`, `instance`, `reward`, `evaluation`, `video_blob`, `model_blob`) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
        cursor.execute(query, (session_id, episode_nr, instance, float(reward), 1 if evaluation else 0, video_blob, model_blob))

        self.db.commit()

        return cursor.lastrowid

    def _cursor(self):
        return self.db.cursor()

    def close(self):
        self.db.close()
