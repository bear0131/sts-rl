"""RunMap: a thin read-only wrapper over the sim's SpireMap for whole-run play.

Why this exists
---------------
SimGameEnv drives a whole run through GameContext, but the gc does NOT expose
"from the node I'm on, which next-row nodes can I move to?" — only
transition_to_map_node(x) to actually move. The standalone sts.SpireMap class
(constructed from the same seed/ascension/act) DOES expose edges + room types,
and a probe confirmed it reconstructs the SAME map the gc walks: SpireMap's
has_edge / get_room_type agree with where gc.transition_to_map_node actually
lands. So we rebuild a SpireMap alongside the gc purely to enumerate legal map
moves. This keeps the sim a black box — no binding changes.

Map geometry (sim Map.h): nodes are a 15-row (y=0..14) x 7-col (x=0..6) grid.
Row 0 is the first room you pick; row 14's edges point at the act boss. Not
every (x,y) holds a room — get_room_type returns NONE for empty slots.

has_edge quirk (bindings-util.cpp): has_edge(-1, _, x2) returns "does first-row
node x2 exist" — exactly the first-move case where the gc's current node is
(-1,-1) before you enter the map. We lean on that so reachable() handles the
first move and every later move uniformly.
"""

from . import _BUILD  # noqa: F401  (ensures sys.path bootstrap ran)
import slaythespire as sts

_NONE = "NONE"
_INVALID = "INVALID"

# rows 0..14; row 14's edges lead to the boss node.
MAX_Y = 14
N_COLS = 7


def _room_name(room):
    """sts.Room enum -> bare name string ('MONSTER', 'SHOP', 'NONE', ...)."""
    return str(room).split(".")[-1]


class RunMap:
    """Reachability/room queries for one act's map, rebuilt from the seed.

    Construct to match the gc: same seed + ascension + act. `burning` controls
    whether a burning (emerald-key) elite is assigned; SimGameEnv matches the
    live green-key flag. Act 4 uses the fixed rest/shop/elite/boss map."""

    def __init__(self, seed, ascension, act, burning=False):
        self.seed = seed
        self.ascension = ascension
        self.act = act
        self._m = sts.SpireMap(seed, ascension, act, burning)

    def reachable(self, x, y):
        """Columns x2 reachable from node (x, y) on the next row.

        For the first move pass (x, y) = (-1, -1): has_edge(-1, _, x2) reports
        whether first-row node x2 exists, so this returns the openable first-row
        columns. boss row (y == MAX_Y): edges point at the boss; the gc accepts
        transition_to_map_node(any x), so we surface the current column."""
        if y >= MAX_Y:
            # boss row: any x works for transition; surface current col (or 0).
            return [x if x >= 0 else 0]
        return [x2 for x2 in range(N_COLS) if self._m.has_edge(x, y, x2)]

    def room_type(self, x, y):
        """Bare room-name string at (x, y); 'NONE' for an empty slot, 'INVALID'
        out of range."""
        return _room_name(self._m.get_room_type(x, y))

    def full_view(self, current_x, current_y):
        """Return the visible act map as a fixed-grid-friendly DAG description.

        Each node carries its outgoing child columns plus three positional
        flags. ``reachable`` means the node lies downstream of the player's
        current position; ``choice`` marks a legal next map choice. The game
        displays the complete act map, so this contains no hidden information.
        """
        if current_y >= MAX_Y:
            # The action after row 14 enters the act boss.  MAP_NODE still uses
            # the current column, so expose that row-14 node as the DAG choice
            # even though there is no row-15 map node to encode.
            choices = {(current_x, current_y)} if current_x >= 0 else set()
        else:
            choices = {
                (x, 0 if current_y < 0 else current_y + 1)
                for x in self.reachable(current_x, current_y)
            }
        downstream = set(choices)
        frontier = list(choices)
        while frontier:
            x, y = frontier.pop()
            if y >= MAX_Y:
                continue
            for child_x in self.reachable(x, y):
                child = (child_x, y + 1)
                if child not in downstream:
                    downstream.add(child)
                    frontier.append(child)

        nodes = []
        for y in range(MAX_Y + 1):
            for x in range(N_COLS):
                room = self.room_type(x, y)
                if room == _NONE:
                    continue
                children = self.reachable(x, y) if y < MAX_Y else []
                nodes.append({
                    "x": x,
                    "y": y,
                    "room": room,
                    "children": children,
                    "current": x == current_x and y == current_y,
                    "choice": (x, y) in choices,
                    "reachable": (x, y) in downstream,
                })
        return nodes

    def __repr__(self):
        return repr(self._m)
