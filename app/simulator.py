"""
Ridi simulaci: vozidla, semafory, cas
"""

import logging

from app.config import MX_ST, err
from app.pathfinding import PF
from app.traffic import TM, VS

log = logging.getLogger(__name__)

PathFinder = PF
TrafficManager = TM
VehicleScheduler = VS
SIMULATION_MAX_STEPS = MX_ST


class Simulator:
    def __init__(self, map_parser, vehicles_data):
        self.map = map_parser
        self.vehicles_data = vehicles_data

        self.traffic = TrafficManager(map_parser, vehicles_data)
        self.scheduler = VehicleScheduler(map_parser, self.traffic, vehicles_data)
        self.pathfinder = PathFinder(map_parser)

        # plan vozidel
        for v in self.scheduler.vehicles:
            if not self.scheduler.schedule_vehicle(v, self.pathfinder):
                log.error(f"V: {v['id']}")

        self.current_step = 0
        self.history = []

    def step(self):
        """Jeden krok simulace"""
        self.current_step += 1

        # aktualizace semaforu
        self.traffic.step(self.current_step)

        # vozdila
        step_state = {"step": self.current_step, "vehicles": {}, "intersections": {}}

        for v in self.scheduler.vehicles:
            pos = self.scheduler.update_vehicle(v, self.current_step, self.traffic)

            step_state["vehicles"][v["id"]] = {
                "pos": pos,
                "status": v["status"],
                "priority": v["priority"],
            }

        # semafor status zaznam
        for iid in self.traffic.intersections_state:
            state = self.traffic.intersections_state[iid]

            step_state["intersections"][iid] = {
                "green": state["cgreen"],
                "pos": state["pos"],
            }

        self.history.append(step_state)
        return True

    def run(self, max_steps=None):
        """simulace run"""
        if max_steps is None:
            max_steps = SIMULATION_MAX_STEPS

        all_arrived = False
        for _ in range(max_steps):
            self.step()

            # zda je dosazen cil
            all_arrived = all(v["status"] == "arrived" for v in self.scheduler.vehicles)

            if all_arrived:
                break

        return self.history

    def get_vehicle_route(self, vehicle_id):
        """cesta vozidel zaznam"""
        route = []
        for step_state in self.history:
            if vehicle_id in step_state["vehicles"]:
                pos = step_state["vehicles"][vehicle_id]["pos"]
                if pos:
                    route.append([step_state["step"], pos[0], pos[1]])
        return route

    def get_intersection_events(self):
        """zaznam semaforu"""
        events = []
        for step_state in self.history:
            for iid, light_state in step_state["intersections"].items():
                if light_state["green"]:
                    events.append(
                        {
                            "step": step_state["step"],
                            "id": iid,
                            "passing": light_state["green"],
                        }
                    )

        dedup = {}
        for e in events:
            key = (e["step"], e["id"])
            if key not in dedup:
                dedup[key] = e

        return list(dedup.values())

    def get_protocol(self, session_id):
        """final protokol"""
        protocol = {"sessionId": session_id, "cars": [], "intersectionsEvents": []}

        # cesty vozidel
        for v in self.scheduler.vehicles:
            route = self.get_vehicle_route(v["id"])
            protocol["cars"].append({"carId": v["id"], "route": route})

        # stavy semaforu
        protocol["intersectionsEvents"] = self.get_intersection_events()

        return protocol
