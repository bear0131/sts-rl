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
    whether a burning (emerald-key) elite is assigned; the gc's own map uses the
    save's setting — for act 1 the default is False and the elite buff is applied
    via node coordinates instead (see SimGameEnv / replay._sync_gc)."""

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

    def __repr__(self):
        return repr(self._m)
