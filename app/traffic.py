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
    def step(s, sn):
        """Compatibility wrapper called by Simulator: advance signals for step `sn`."""
        s.st(sn)

    @property
    def intersections_state(s):
        """Return a mapping of intersection id -> current state expected by Simulator.

        Each value contains keys `cgreen` (current green direction) and `pos` (position tuple).
        """
        state = {}
        for iid, info in s.ist.items():
            state[iid] = {"cgreen": info.get("cg"), "pos": info.get("p")}
        return state
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
        s.vehicles = []

        for vehicle in vd:
            s.vehicles.append(
                {
                    "id": vehicle["carId"],
                    "type": vehicle.get("type"),
                    "from": tuple(vehicle["from"]),
                    "to": tuple(vehicle["to"]),
                    "priority": vehicle.get("priority", 0),
                    "start_time": vehicle.get("startTime", 0),
                    "path": None,
                    "current_pos": None,
                    "status": "waiting",
                    "waiting_at": None,
                }
            )

        s.vehicles.sort(key=lambda v: v["priority"])

    def schedule_vehicle(s, vehicle, pf):
        path = pf.find_path(vehicle["from"], vehicle["to"])
        if not path:
            return False
        vehicle["path"] = path
        return True

    def update_vehicle(s, vehicle, current_step, traffic_mgr):
        """stav vozidla, vraci pozici nebo None"""

        if current_step < vehicle["start_time"]:
            return None

        if current_step == vehicle["start_time"]:
            vehicle["status"] = "driving"
            vehicle["current_pos"] = vehicle["from"]
            return vehicle["from"]

        if not vehicle["path"]:
            return None

        steps_traveled = current_step - vehicle["start_time"]
        path_idx = min(steps_traveled, len(vehicle["path"]) - 1)

        if path_idx >= len(vehicle["path"]) - 1:
            vehicle["status"] = "arrived"
            vehicle["current_pos"] = vehicle["path"][-1]
            return vehicle["path"][-1]

        next_pos = vehicle["path"][path_idx]

        if not traffic_mgr.mp.is_passable(*next_pos):
            vehicle["status"] = "blocked"
            vehicle["waiting_at"] = next_pos
            return vehicle["current_pos"]

        if not traffic_mgr.cp(next_pos[0], next_pos[1], vehicle["priority"]):
            vehicle["status"] = "waiting"
            vehicle["waiting_at"] = next_pos
            return vehicle["current_pos"]

        vehicle["status"] = "driving"
        vehicle["waiting_at"] = None
        vehicle["current_pos"] = next_pos
        return next_pos
