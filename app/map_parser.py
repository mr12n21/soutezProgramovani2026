"""
Parser mapy
https://docs.python.org/3/library/array.html
https://pillow.readthedocs.io/
"""

import numpy as np
from PIL import Image


class MP:
    """parsovani mapy"""

    def __init__(self, img, bs=10):
        self.img = img
        self.bs = bs
        self.d = np.array(img)
        self.h, self.w = self.d.shape[:2]
        self.bh = self.h // bs
        self.bw = self.w // bs
        self.rd = set()
        self.lt = {}
        self.jnc = []
        self.jiid = {}
        self._sc()

    def _wh(self, p):
        """bila"""
        return p[0] > 200 and p[1] > 200 and p[2] > 200

    def _rd(self, p):
        """cervena"""
        return p[0] > 150 and p[1] < 100 and p[2] < 100

    def _gr(self, p):
        """zelena"""
        return p[1] > 150 and p[0] < 100 and p[2] < 100

    def _sc(self):
        """cesty a krizovatky"""
        for by in range(self.bh):
            for bx in range(self.bw):
                px = self.d[by * self.bs, bx * self.bs]
                if self._wh(px):
                    self.rd.add((bx, by))

        jm = {}
        for bx, by in self.rd:
            n = 0
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = bx + dx, by + dy
                if 0 <= nx < self.bw and 0 <= ny < self.bh:
                    if (nx, ny) in self.rd:
                        n += 1
            if n >= 3:
                jm[(bx, by)] = n

        self.jnc = sorted(jm.keys(), key=lambda k: (k[1], k[0]))
        self.jiid = {p: f"K{i + 1}" for i, p in enumerate(self.jnc)}

        # semafory
        for by in range(self.bh):
            for bx in range(self.bw):
                px = self.d[by * self.bs, bx * self.bs]

                if self._rd(px) or self._gr(px):
                    st = "red" if self._rd(px) else "green"
                    self.lt[f"{bx}_{by}"] = {"st": st, "p": (bx, by)}

    def pp(self, x, y):
        """jizda"""
        return 0 <= x < self.bw and 0 <= y < self.bh and (x, y) in self.rd

    def gi(self, x, y):
        """krizovka"""
        return self.jiid.get((x, y))

    def gip(self, i):
        """krizovatka dle indexu"""
        return self.jnc[i] if i < len(self.jnc) else None

    @property
    def intersection_ids(self):
        return self.jiid

    @property
    def lights(self):
        return bool(self.lt)

    @property
    def intersections(self):
        return self.jnc

    @property
    def roads(self):
        return self.rd

    def get_intersection_id(self, x, y):
        return self.gi(x, y)

    def is_passable(self, x, y):
        return self.pp(x, y)
