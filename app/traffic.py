"""
semafory a priority
"""

from app.config import err


class TM:
    """ridic semaforu"""

    def __init__(s, mp, vh):
        s.mp = mp
        s.vh = vh
        s.ist = {}

        for p, iid in mp.intersection_ids.items():
            s.ist[iid] = {
                "p": p,
                "cg": None,
                "st": 0,
                "hl": bool(mp.lights),
                "r": ["S", "V", "J", "Z"],
            }

    def st(s, sn):
        """status semafor"""
        for iid, st in s.ist.items():
            if not st["hl"]:
                continue

            if st["cg"] is None:
                st["cg"] = st["r"][0]
            else:
                idx = st["r"].index(st["cg"])
                st["cg"] = st["r"][(idx + 1) % len(st["r"])]

            st["st"] = sn

    def cp(s, x, y, pr):
        """prujezd krizovatkou"""
        iid = s.mp.get_intersection_id(x, y)
        if not iid:
            return True

        st = s.ist.get(iid)
        if not st or not st["hl"]:
            return True

        dr = s._gd(x, y, iid)

        if st["cg"] == dr:
            return True

        return pr <= 1

    def _gd(s, x, y, iid):
        """misto prijezdu na krizovatku"""
        kx, ky = s.ist[iid]["p"]
        dx = abs(x - kx)
        dy = abs(y - ky)

        if dy <= dx:
            return "V" if x > kx else "Z"
        else:
            return "J" if y > ky else "S"

    def gs(s, iid, st):
        """barva v korku"""
        st = s.ist.get(iid)
        if not st or not st["hl"]:
            return ""
        idx = (st // 1) % 4
        return st["r"][idx]


class VS:
    """plan: priority, misto jizdy"""

    def __init__(s, mp, tm, vd):
        s.mp = mp
        s.tm = tm
        s.v = []

        for v in vd:
            s.v.append(
                {
                    "id": v["carId"],
                    "t": v["type"],
                    "f": tuple(v["from"]),
                    "t_o": tuple(v["to"]),
                    "pr": v["priority"],
                    "st": v["startTime"],
                    "p": None,
                    "cp": None,
                    "st_v": "waiting",
                    "wa": None,
                }
            )

        s.v.sort(key=lambda v: v["pr"])

    def sv(s, v, pf):
        p = pf.fp(v["f"], v["t_o"])
        if not p:
            return err(f"Auto: {v['id']}")
        v["p"] = p
        return True

    def uv(s, v, st, tm):
        """moznost pohybu"""
        if st < v["st"]:
            return None
        if st == v["st"]:
            v["st_v"], v["cp"] = "driving", v["f"]
            return v["f"]
        if not v["p"]:
            return None

        idx = min(st - v["st"], len(v["p"]) - 1)
        if idx >= len(v["p"]) - 1:
            v["st_v"] = "arrived"
            return v["p"][-1]

        nx = v["p"][idx]

        if not tm.mp.is_passable(*nx):
            v["st_v"], v["wa"] = "blocked", nx
        elif not tm.cp(nx[0], nx[1], v["pr"]):
            v["st_v"], v["wa"] = "waiting", nx
        else:
            v["st_v"], v["wa"], v["cp"] = "driving", None, nx

        return v["cp"]

        vehicle["path"] = path
        return True

    def update_vehicle(self, vehicle, current_step, traffic_mgr):
        """
        stav vozidla
        vrazeni pozice nebo jede dal none
        """

        # je na pame
        if current_step < vehicle["start_time"]:
            return None

        # objeveni na mape
        if current_step == vehicle["start_time"]:
            vehicle["status"] = "driving"
            vehicle["current_pos"] = vehicle["from"]
            return vehicle["from"]

        # mapa uz je
        if not vehicle["path"] or len(vehicle["path"]) == 0:
            return None

        # vzdalenost cesty
        steps_traveled = current_step - vehicle["start_time"]
        path_idx = min(steps_traveled, len(vehicle["path"]) - 1)

        # cil detecke
        if path_idx >= len(vehicle["path"]) - 1:
            vehicle["status"] = "arrived"
            return vehicle["path"][-1]

        next_pos = vehicle["path"][path_idx]

        # overeni priority a pravidel
        if not traffic_mgr.map.is_passable(next_pos[0], next_pos[1]):
            vehicle["status"] = "blocked"
            vehicle["waiting_at"] = next_pos
            return vehicle["current_pos"]

        # moznost jet
        can_go = traffic_mgr.can_pass_intersection(
            next_pos[0], next_pos[1], vehicle["priority"]
        )

        if not can_go:
            # ceka
            vehicle["status"] = "waiting"
            vehicle["waiting_at"] = next_pos
            return vehicle["current_pos"]

        vehicle["status"] = "driving"
        vehicle["waiting_at"] = None
        vehicle["current_pos"] = next_pos
        return next_pos
