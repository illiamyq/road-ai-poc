import numpy as np
from highway_env.road.lane import CircularLane, LineType, SineLane, StraightLane
from highway_env.road.road import RoadNetwork

BUILDERS = {}
WIDTH = StraightLane.DEFAULT_WIDTH
CONT, STRIPED, NONE = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE


def builder(name):
    def wrap(fn):
        BUILDERS[name] = fn
        return fn
    return wrap


class Cursor:
    def __init__(self, lanes, speed_limit):
        self.net = RoadNetwork()
        self.lanes = lanes
        self.speed_limit = speed_limit
        self.x = 0.0
        self.node = "n0"
        self.count = 0
        self.corridor = []
        self.crossings = []
        self.ramps = []
        self.obstacles = []
        self.ego = None
        self.merge = None
        self.exit = None

    def next_node(self, prefix="n"):
        self.count += 1
        return f"{prefix}{self.count}"

    def add_highway(self, length, lanes=None):
        end = self.next_node()
        RoadNetwork.straight_road_network(
            lanes or self.lanes,
            start=self.x,
            length=length,
            nodes_str=(self.node, end),
            speed_limit=self.speed_limit,
            net=self.net,
        )
        if self.ego is None:
            self.ego = (self.node, end, None, 10.0)
        self.corridor.append((self.node, end, self.x, self.x + length))
        self.node, self.x = end, self.x + length
        return end


@builder("straight")
def straight(cur, length):
    cur.add_highway(length)


@builder("ramp")
def ramp(cur, before=150.0, converge=80.0, parallel=80.0, ego_start=30.0):
    # ? before 90, conv 40, ?
    # rem to test in custom env, set vehicles_count=10,100, ramp_vehicles=0,2,5
    x0 = cur.x
    j, k = cur.next_node("rj"), cur.next_node("rk")
    if cur.ego is None:
        cur.ego = (j, k, 0, ego_start)
    b = cur.add_highway(before + converge)
    c = cur.add_highway(parallel)
    amp = 3.25
    y_par = cur.lanes * WIDTH
    y_app = y_par + 2 * amp
    ljk = StraightLane(
        [x0, y_app], [x0 + before, y_app],
        line_types=[CONT, CONT], forbidden=True, speed_limit=cur.speed_limit,
    )
    lkb = SineLane(
        [x0 + before, y_par + amp], [x0 + before + converge, y_par + amp],
        amp, 2 * np.pi / (2 * converge), np.pi / 2,
        line_types=[CONT, CONT], forbidden=True, speed_limit=cur.speed_limit,
    )
    lbc = StraightLane(
        [x0 + before + converge, y_par], [x0 + before + converge + parallel, y_par],
        line_types=[STRIPED, CONT], forbidden=True, speed_limit=cur.speed_limit,
    )
    cur.net.add_lane(j, k, ljk)
    cur.net.add_lane(k, b, lkb)
    cur.net.add_lane(b, c, lbc)
    cur.obstacles.append(lbc.position(parallel, 0.0))
    cur.ramps.append((j, k, before))
    cur.merge = (b, c, cur.lanes)


@builder("exit")
def exit_zone(cur, length=150.0, radius=150.0):
    # more interesting when on left // current policy on the right though to test more
    a = cur.node
    b = cur.add_highway(length, lanes=cur.lanes + 1)
    node = cur.next_node("exit")
    center = np.array([cur.x, cur.lanes * WIDTH + radius])
    lane = CircularLane(
        center=center, radius=radius,
        start_phase=3 * np.pi / 2, end_phase=2 * np.pi, forbidden=True,
    )
    cur.net.add_lane(b, node, lane)
    cur.exit = {"from": a, "to": b, "lane": (a, b, cur.lanes), "node": node, "end_x": cur.x}


@builder("crossing")
def crossing(cur, box=20.0, side_length=120.0):
    start_x = cur.x
    cur.add_highway(box)
    k = len(cur.crossings)
    xc = start_x + box / 2
    mid = (cur.lanes - 1) * WIDTH / 2
    lane = StraightLane(
        [xc, mid - side_length], [xc, mid + side_length],
        line_types=[CONT, CONT], speed_limit=cur.speed_limit,
    )
    a, b = f"x{k}_a", f"x{k}_b"
    cur.net.add_lane(a, b, lane)
    cur.crossings.append((a, b, 2 * side_length, xc))


def merge_exit_layout(before=150.0, converge=80.0, parallel=80.0, cruise=400.0, exit_length=150.0, tail=200.0):
    return [
        {"type": "ramp", "before": before, "converge": converge, "parallel": parallel},
        {"type": "straight", "length": cruise},
        {"type": "exit", "length": exit_length},
        {"type": "straight", "length": tail},
    ]


def corridor_layout(crossings=5, approach=150.0, box=20.0, side_length=120.0):
    parts = []
    for _ in range(crossings):
        parts.append({"type": "straight", "length": approach})
        parts.append({"type": "crossing", "box": box, "side_length": side_length})
    parts.append({"type": "straight", "length": approach})
    return parts