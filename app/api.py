import json
import logging
from io import BytesIO

import requests
from PIL import Image

log = logging.getLogger(__name__)


class APIClient:
    def __init__(self, bUrl, aKey):
        self.base = bUrl.rstrip("/")
        self.key = aKey
        self.sid = None
        self.vehicles = []

    @property
    def session_id(self):
        return self.sid

    """
    https://stackoverflow.com/questions/21965484/timeout-for-python-requests-get-entire-response
    """

    def get_settings(self, map_name=None):
        # get ssid
        params = {"apiKey": self.key}
        if map_name:
            params["map"] = map_name
        try:
            """
            https://stackoverflow.com/questions/42601812/python-requests-url-base-in-session
            https://www.geeksforgeeks.org/python/response-raise_for_status-python-requests/
            """
            r = requests.get(f"{self.base}/settings", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            self.sid = data["sessionId"]
            self.vehicles = data.get("vehicles", [])
            return data
        except Exception as e:
            log.error(f"GET: {e}")
            raise

    def get_map(self):
        """stazeni mapy"""
        if not self.sid:
            raise ValueError("E:get_map")

        params = {"sessionId": self.sid}

        try:
            r = requests.get(f"{self.base}/map", params=params, timeout=10)
            r.raise_for_status()

            img = Image.open(BytesIO(r.content))
            return img
        except Exception as e:
            log.error(f"GET: {e}")
            raise

    def get_requests(self):
        """seznam vozidel na simulaci"""
        if not self.sid:
            raise ValueError("get_settings()")

        params = {"sessionId": self.sid}

        try:
            r = requests.get(f"{self.base}/requests", params=params, timeout=10)
            r.raise_for_status()

            reqs = r.json()
            return reqs
        except Exception as e:
            log.error(f"GET: {e}")
            raise

    def post_protocol(self, protocol_data):
        """POST protokol"""
        if not self.sid:
            raise ValueError("SessionId chybí!")

        payload = {
            "sessionId": self.sid,
            **protocol_data,
        }

        try:
            r = requests.post(f"{self.base}/protocol", json=payload, timeout=10)
            r.raise_for_status()

            result = r.json() if r.content else {}
            return result
        except Exception as e:
            log.error(f"POST: {e}")
            raise
