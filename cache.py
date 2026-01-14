import time, os
import uuid


class Cache:
    def __init__(self, base_path = '/srv/share/dsa_eval_chat_session'):
        self.base_path = base_path
        self.session_cache_path = self.create_session_path()

    def create_session_path(self):
        current_date = time.strftime('%Y-%m-%d', time.localtime())

        while True:
            new_uuid = uuid.uuid4().hex
            folder_name = f"{current_date}-{new_uuid}"
            session_cache_path = os.path.join(self.base_path, folder_name)

            if not os.path.exists(session_cache_path):
                os.mkdir(session_cache_path)
                return session_cache_path
