import os

K = os.getenv("API_KEY")
U = os.getenv("SERVER_URL")

BLK = 10
MX_ST = 500
DLY = 0.1


def err(message):
    raise ValueError(message)

LR = 3
LP = 1

W = 1.0
