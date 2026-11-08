"""RunMap: a thin read-only wrapper over the sim's SpireMap for whole-run play.

Why this exists
---------------
SimGameEnv obtains live legal choices from GameContext. The standalone sts.SpireMap class
(constructed from the same seed/ascension/act) DOES expose edges + room types,
and a probe confirmed it reconstructs the SAME map the gc walks: SpireMap's
has_edge / get_room_type agree with where gc.transition_to_map_node actually
lands. So we rebuild a SpireMap alongside the gc purely to enumerate legal map
moves for the physical DAG. Relic-assisted choices come from the live native
legal_map_options, not this static topology wrapper.

Map geometry (sim Map.h): nodes are a 15-row (y=0..14) x 7-col (x=0..6) grid.
Row 0 is the first room you pick; row 14's edges point at the act boss. Not
every (x,y) holds a room — get_room_type returns NONE for empty slots.

has_edge quirk (bindings-util.cpp): has_edge(-1, _, x2) returns "does first-row
node x2 exist" — exactly the first-move case where the gc's current node is
(-1,-1) before you enter the map. We lean on that so reachable() handles the
first move and every later move uniformly.
"""

from functools import lru_cache

from . import _BUILD  # noqa: F401  (ensures sys.path bootstrap ran)
import slaythespire as sts

_NONE = "NONE"
_INVALID = "INVALID"

# rows 0..14; row 14's edges lead to the boss node.
MAX_Y = 14
N_COLS = 7


@lru_cache(maxsize=32)
def _room_name_value(value):
    return str(sts.Room(value)).split(".")[-1]


def _room_name(room):
    """Cache an owned integer, never a possibly borrowed mutable enum."""
    return _room_name_value(int(room))


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
        columns. boss row (y == MAX_Y): edges point at the boss; surface the
        current column, matching the native action codec."""
        if y >= MAX_Y:
            if y != MAX_Y or not 0 <= x < N_COLS:
                raise ValueError('boss map choice needs its actual final-row coordinate')
            return [x]
        return [x2 for x2 in range(N_COLS) if self._m.has_edge(x, y, x2)]

    def room_type(self, x, y):
        """Bare room-name string at (x, y); 'NONE' for an empty slot, 'INVALID'
        out of range."""
        return _room_name(self._m.get_room_type(x, y))

    def full_view(self, current_x, current_y, *, next_columns=None):
        """Return the visible act map as a fixed-grid-friendly DAG description.

        Each node carries its outgoing child columns plus three positional
        flags. ``reachable`` means the node lies downstream of the player's
        current position; ``choice`` marks a legal next map choice. The game
        displays the complete act map, so this contains no hidden information.
        ``next_columns`` comes from the live GameContext (including Wing Boots).
        Children remain physical map edges, not hypothetical relic-assisted jumps.
        """
        # SpireMap is read-only. Retain only its physical topology, not the
        # live choices (Wing Boots) or player-position flags. Keep each cache
        # on its map owner so a new act/reset cannot reuse another map's DAG.
        topology = getattr(self, '_full_view_topology', None)
        if topology is None or topology[0] is not self._m:
            nodes, edges = [], {}
            for y in range(MAX_Y + 1):
                for x in range(N_COLS):
                    room = self.room_type(x, y)
                    if room == _NONE:
                        continue
                    children = tuple(self.reachable(x, y)) if y < MAX_Y else ()
                    nodes.append((x, y, room, children))
                    edges[x, y] = children
            topology = self._m, tuple(nodes), edges
            self._full_view_topology = topology
        _, nodes, edges = topology
        if current_y >= MAX_Y:
            # The action after row 14 enters the act boss.  MAP_NODE still uses
            # the current column, so expose that row-14 node as the DAG choice
            # even though there is no row-15 map node to encode.
            choices = {(current_x, current_y)} if current_x >= 0 else set()
        else:
            choices = {
                (x, 0 if current_y < 0 else current_y + 1)
                for x in (self.reachable(current_x, current_y)
                          if next_columns is None else next_columns)
            }
        downstream = set(choices)
        frontier = list(choices)
        while frontier:
            x, y = frontier.pop()
            if y >= MAX_Y:
                continue
            for child_x in edges.get((x, y), ()):
                child = (child_x, y + 1)
                if child not in downstream:
                    downstream.add(child)
                    frontier.append(child)

        return [{
            "x": x,
            "y": y,
            "room": room,
            "children": list(children),
            "current": x == current_x and y == current_y,
            "choice": (x, y) in choices,
            "reachable": (x, y) in downstream,
        } for x, y, room, children in nodes]

    def __repr__(self):
        return repr(self._m)

    def encoded_view(self, current_x, current_y, *, next_columns, global_map_view):
        """One validated projection, reused only for identical live map inputs.

        The native topology is immutable. Legal columns are supplied afresh by
        GameContext, including Wing Boots, and their order affects the preview.
        Returned arrays are internal read-only templates; encode copies them.
        """
        from . import encode

        columns = tuple(next_columns)
        key = current_x, current_y, columns, global_map_view
        cached = getattr(self, '_encoded_view', None)
        if cached is None or cached[0] is not self._m or cached[1] != key:
            options = dict(nodes=[(x, 'BOSS' if current_y == MAX_Y else
                                  self.room_type(x, max(0, current_y + 1)))
                                 for x in columns])
            if (global_map_view and type(current_x) is int and type(current_y) is int
                    and -1 <= current_y <= MAX_Y):
                fields = self._project_numeric_map(current_x, current_y, options['nodes'])
                cached = self._m, key, fields
                self._encoded_view = cached
                return fields
            if global_map_view:
                options['full_map'] = self.full_view(
                    current_x, current_y, next_columns=columns)
            empty = encode._empty(global_map_view=global_map_view)
            fields = {name: value for name, value in empty.items()
                      if name.startswith('map_')}
            # Preserve one codec and all its validation, including complete
            # candidates, the frozen three-token preview and boss identity.
            encode._encode_options(fields, dict(phase='map', options=options))
            for value in fields.values():
                value.setflags(write=False)
            cached = self._m, key, fields
            self._encoded_view = cached
        return cached[2]


    def _project_numeric_map(self, current_x, current_y, nodes):
        from . import encode
        import numpy as np

        # The sole codec validates both topology and complete live candidates.
        # Only immutable topology survives a position/choice change.
        choices = encode._map_choice_ids(nodes, encode.MAP_COLS)
        cached = getattr(self, '_numeric_map_topology', None)
        if cached is None or cached[0] is not self._m:
            graph = self.full_view(-1, -1, next_columns=[])
            empty = encode._empty(global_map_view=True)
            fields = {name: value for name, value in empty.items() if name.startswith('map_')}
            encode._encode_options(fields, dict(phase='map', options=dict(nodes=[], full_map=graph)))
            descendants = np.zeros((encode.MAP_FULL_SLOTS, encode.MAP_FULL_SLOTS), dtype=bool)
            for node in sorted(graph, key=lambda n: n['y'], reverse=True):
                slot = node['y'] * encode.MAP_COLS + node['x']
                descendants[slot, slot] = True
                for child in node['children']:
                    descendants[slot] |= descendants[(node['y'] + 1) * encode.MAP_COLS + child]
            for value in fields.values():
                value.setflags(write=False)
            descendants.setflags(write=False)
            cached = self._m, fields, descendants
            self._numeric_map_topology = cached
        _, template, descendants = cached
        fields = {key: value.copy() for key, value in template.items()}
        features = fields['map_full_features']
        rooms = fields['map_full_rooms']
        if 0 <= current_x < encode.MAP_COLS and 0 <= current_y < encode.MAP_ROWS:
            current_slot = current_y * encode.MAP_COLS + current_x
            if rooms[current_slot]:
                features[current_slot, 2] = 1.
        if current_y == MAX_Y:
            expected = {current_x} if current_x >= 0 else set()
            if set(choices) != expected:
                raise ValueError("full-map choices do not cover the complete action candidates")
        row = max(0, min(MAX_Y, current_y + 1))
        slots = [row * encode.MAP_COLS + x for x in choices]
        for (x, room), slot in zip(nodes, slots, strict=True):
            if not rooms[slot]:
                raise ValueError("full-map choices do not cover the complete action candidates")
            boss = room == 'BOSS' and len(nodes) == 1 and current_y == MAX_Y and x == current_x
            if not boss and rooms[slot] != choices[x]:
                raise ValueError("map candidate room differs from full-map room")
        if slots:
            features[slots, 3] = 1.
            features[np.any(descendants[slots], axis=0), 4] = 1.
        # Keep the frozen three-slot preview in the same codec. Capacity was
        # checked against all seven live columns above, not this preview.
        preview = {key: fields[key] for key in ('map_node_rooms', 'map_node_x')}
        encode._encode_options(preview, dict(phase='map', options=dict(nodes=nodes[:encode.MAP_EDGES])))
        for value in fields.values():
            value.setflags(write=False)
        return fields
