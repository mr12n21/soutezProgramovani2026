"""
A* algoritmus pro hledani cesty
https://en.wikipedia.org/wiki/A*_search_algorithm
https://docs.python.org/3/library/heapq.html
"""

import heapq

from app.config import err


class PF:
    def __init__(s, mp):
        s.mp = mp

    def fp(s, st, gl):
        """start > cil"""
        if not s.mp.is_passable(*st) or not s.mp.is_passable(*gl):
            return

        if st == gl:
            return [st]

        op = [(0, st)]
        cf = {}
        g = {st: 0}
        h = {st: s._h(st, gl)}
        cl = set()

        while op:
            _, c = heapq.heappop(op)

            if c in cl:
                continue

            if c == gl:
                return s._rp(cf, c)

            cl.add(c)

            # 4 smery sousede
            for nx, ny in [
                (c[0] + dx, c[1] + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
            ]:
                if not s.mp.is_passable(nx, ny) or (nx, ny) in cl:
                    continue

                tg = g[c] + 1

                if (nx, ny) not in g or tg < g[(nx, ny)]:
                    cf[(nx, ny)] = c
                    g[(nx, ny)] = tg
                    hn = s._h((nx, ny), gl)
                    h[(nx, ny)] = hn
                    fn = tg + hn
                    heapq.heappush(op, (fn, (nx, ny)))

        return

    def _h(s, a, b):
        """Manhattan distance zdroj: https://www.geeksforgeeks.org/data-science/manhattan-distance/"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _rp(s, cf, c):
        """rekonstrukce od cile do startu"""
        p = [c]
        while c in cf:
            c = cf[c]
            p.append(c)
        return p[::-1]

    def find_path(s, st, gl):
        return s.fp(st, gl)
