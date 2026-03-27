"""Hex grid board for the 16x17 Woodland map.

Coordinate system: odd-column offset (same as MegaMek).
  - x = column (0-15), y = row (0-16)
  - Odd columns are shifted down by half a hex.
  - 6 directions: 0=N, 1=NE, 2=SE, 3=S, 4=SW, 5=NW
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


class Terrain(IntEnum):
    CLEAR = 0
    LIGHT_WOODS = 1
    HEAVY_WOODS = 2
    WATER = 3       # Impassable for ground mechs
    ROUGH = 4       # +1 MP to enter


class HexData(NamedTuple):
    elevation: int
    terrain: Terrain


WIDTH = 16
HEIGHT = 17


def _parse_board() -> dict[tuple[int, int], HexData]:
    """Build board data from the Map Set 6/16x17 Woodland board file.

    Hex IDs in the .board file are 1-based XXYY; we store 0-based (x, y).
    """
    hexes: dict[tuple[int, int], HexData] = {}
    for line in _BOARD_DATA.strip().splitlines():
        parts = line.split()
        # Format: XXYY elevation "terrain"
        hex_id = parts[0]
        elev = int(parts[1])
        terrain_str = parts[2] if len(parts) > 2 else ""

        col = int(hex_id[:2]) - 1  # 0-based x
        row = int(hex_id[2:]) - 1  # 0-based y

        if "woods:2" in terrain_str:
            terrain = Terrain.HEAVY_WOODS
        elif "woods:1" in terrain_str:
            terrain = Terrain.LIGHT_WOODS
        elif "water" in terrain_str:
            terrain = Terrain.WATER
        elif "rough" in terrain_str:
            terrain = Terrain.ROUGH
        else:
            terrain = Terrain.CLEAR

        hexes[(col, row)] = HexData(elev, terrain)
    return hexes


# Direction offsets for odd-column offset hex grid.
# For even columns (x % 2 == 0): neighbors shift
# For odd columns (x % 2 == 1): neighbors shift differently
_EVEN_COL_DIRS = [
    (0, -1),   # 0: N
    (1, -1),   # 1: NE
    (1, 0),    # 2: SE
    (0, 1),    # 3: S
    (-1, 0),   # 4: SW
    (-1, -1),  # 5: NW
]

_ODD_COL_DIRS = [
    (0, -1),   # 0: N
    (1, 0),    # 1: NE
    (1, 1),    # 2: SE
    (0, 1),    # 3: S
    (-1, 1),   # 4: SW
    (-1, 0),   # 5: NW
]


def neighbor(x: int, y: int, direction: int) -> tuple[int, int]:
    """Get the neighbor hex in the given direction (0-5)."""
    if x % 2 == 0:
        dx, dy = _EVEN_COL_DIRS[direction]
    else:
        dx, dy = _ODD_COL_DIRS[direction]
    return x + dx, y + dy


def neighbors(x: int, y: int) -> list[tuple[int, int]]:
    """Get all valid neighbor hexes."""
    result = []
    for d in range(6):
        nx, ny = neighbor(x, y, d)
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            result.append((nx, ny))
    return result


def neighbors_with_dir(x: int, y: int) -> list[tuple[int, int, int]]:
    """Get all valid neighbor hexes with their direction."""
    result = []
    for d in range(6):
        nx, ny = neighbor(x, y, d)
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            result.append((nx, ny, d))
    return result


class Board:
    """Static hex board with terrain and elevation."""

    def __init__(self) -> None:
        self.width = WIDTH
        self.height = HEIGHT
        self._hexes = _parse_board()

    def get(self, x: int, y: int) -> HexData:
        return self._hexes.get((x, y), HexData(0, Terrain.CLEAR))

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def elevation(self, x: int, y: int) -> int:
        return self.get(x, y).elevation

    def terrain(self, x: int, y: int) -> Terrain:
        return self.get(x, y).terrain

    def movement_cost(self, x: int, y: int) -> int:
        """Extra MP cost for terrain in this hex (added to base cost of 1)."""
        t = self.terrain(x, y)
        if t == Terrain.LIGHT_WOODS:
            return 1
        elif t == Terrain.HEAVY_WOODS:
            return 2
        elif t == Terrain.ROUGH:
            return 1
        return 0

    def is_passable(self, x: int, y: int) -> bool:
        """Check if a ground mech can enter this hex."""
        return self.terrain(x, y) != Terrain.WATER

    def to_obs_hexes(self) -> list[dict]:
        """Produce hex list matching Java's ObservationBuilder format."""
        result = []
        for y in range(self.height):
            for x in range(self.width):
                h = self.get(x, y)
                terrain_str = ""
                if h.terrain == Terrain.LIGHT_WOODS:
                    terrain_str = "Light Woods"
                elif h.terrain == Terrain.HEAVY_WOODS:
                    terrain_str = "Heavy Woods"
                elif h.terrain == Terrain.WATER:
                    terrain_str = "Water"
                elif h.terrain == Terrain.ROUGH:
                    terrain_str = "Rough"
                result.append({
                    "x": x,
                    "y": y,
                    "elevation": h.elevation,
                    "terrain": terrain_str,
                })
        return result


# Raw board data extracted from Map Set 6/16x17 Woodland.board
# Format per line: XXYY elevation "terrain"
_BOARD_DATA = """\
0101 0 ""
0201 0 ""
0301 0 ""
0401 0 ""
0501 0 ""
0601 0 ""
0701 0 ""
0801 0 ""
0901 0 ""
1001 0 "woods:1;foliage_elev:2"
1101 0 ""
1201 0 ""
1301 0 ""
1401 0 ""
1501 0 ""
1601 0 ""
0102 0 ""
0202 0 ""
0302 0 ""
0402 0 "woods:1;foliage_elev:2"
0502 0 "woods:1;foliage_elev:2"
0602 0 ""
0702 0 ""
0802 0 ""
0902 0 ""
1002 0 ""
1102 0 "woods:1;foliage_elev:2"
1202 0 "woods:1;foliage_elev:2"
1302 0 ""
1402 0 ""
1502 0 ""
1602 0 ""
0103 0 "woods:1;foliage_elev:2"
0203 0 "woods:1;foliage_elev:2"
0303 0 "woods:2;foliage_elev:2"
0403 1 "woods:2;foliage_elev:2"
0503 1 "woods:1;foliage_elev:2"
0603 1 ""
0703 0 ""
0803 0 ""
0903 0 ""
1003 0 ""
1103 0 "woods:1;foliage_elev:2"
1203 0 "woods:1;foliage_elev:2"
1303 0 ""
1403 0 ""
1503 0 ""
1603 0 ""
0104 0 ""
0204 0 "woods:1;foliage_elev:2"
0304 0 "woods:1;foliage_elev:2"
0404 1 "woods:1;foliage_elev:2"
0504 1 ""
0604 2 ""
0704 1 ""
0804 1 ""
0904 0 ""
1004 0 ""
1104 0 ""
1204 0 ""
1304 0 ""
1404 0 ""
1504 0 ""
1604 0 ""
0105 0 ""
0205 0 "woods:1;foliage_elev:2"
0305 0 "woods:1;foliage_elev:2"
0405 1 ""
0505 2 ""
0605 3 ""
0705 2 ""
0805 1 "woods:1;foliage_elev:2"
0905 0 ""
1005 0 ""
1105 0 ""
1205 0 "woods:1;foliage_elev:2"
1305 0 "woods:1;foliage_elev:2"
1405 0 "woods:1;foliage_elev:2"
1505 0 ""
1605 0 ""
0106 0 "woods:1;foliage_elev:2"
0206 0 "woods:1;foliage_elev:2"
0306 0 ""
0406 1 ""
0506 2 ""
0606 2 "woods:1;foliage_elev:2"
0706 2 "woods:2;foliage_elev:2"
0806 1 "woods:1;foliage_elev:2"
0906 0 "woods:1;foliage_elev:2"
1006 0 ""
1106 0 ""
1206 0 ""
1306 0 ""
1406 0 ""
1506 0 ""
1606 0 ""
0107 0 ""
0207 0 ""
0307 1 ""
0407 1 ""
0507 1 ""
0607 1 ""
0707 1 "woods:1;foliage_elev:2"
0807 0 "woods:1;foliage_elev:2"
0907 0 ""
1007 0 ""
1107 0 ""
1207 0 "woods:2;foliage_elev:2"
1307 0 ""
1407 0 ""
1507 0 ""
1607 0 ""
0108 0 ""
0208 0 ""
0308 1 ""
0408 0 ""
0508 0 ""
0608 0 "woods:1;foliage_elev:2"
0708 0 "woods:1;foliage_elev:2"
0808 0 ""
0908 0 ""
1008 0 ""
1108 0 "woods:1;foliage_elev:2"
1208 1 "woods:1;foliage_elev:2"
1308 1 ""
1408 1 ""
1508 0 ""
1608 0 ""
0109 0 ""
0209 0 ""
0309 0 ""
0409 0 ""
0509 0 ""
0609 0 ""
0709 0 "woods:1;foliage_elev:2"
0809 0 ""
0909 0 ""
1009 0 ""
1109 0 ""
1209 0 ""
1309 1 ""
1409 1 ""
1509 1 ""
1609 0 ""
0110 0 ""
0210 0 "woods:1;foliage_elev:2"
0310 0 ""
0410 0 ""
0510 0 ""
0610 0 ""
0710 0 ""
0810 0 ""
0910 0 ""
1010 0 "woods:1;foliage_elev:2"
1110 0 ""
1210 0 ""
1310 0 ""
1410 0 ""
1510 1 ""
1610 0 ""
0111 0 ""
0211 0 ""
0311 0 "woods:1;foliage_elev:2"
0411 0 "woods:1;foliage_elev:2"
0511 0 ""
0611 1 ""
0711 1 ""
0811 1 ""
0911 0 ""
1011 0 "woods:2;foliage_elev:2"
1111 0 "woods:1;foliage_elev:2"
1211 0 "woods:1;foliage_elev:2"
1311 0 ""
1411 0 ""
1511 0 ""
1611 0 ""
0112 0 ""
0212 0 ""
0312 0 "woods:1;foliage_elev:2"
0412 0 "woods:1;foliage_elev:2"
0512 0 "woods:1;foliage_elev:2"
0612 1 ""
0712 1 ""
0812 2 ""
0912 1 ""
1012 0 "woods:1;foliage_elev:2"
1112 0 "woods:1;foliage_elev:2"
1212 0 ""
1312 0 "woods:1;foliage_elev:2"
1412 0 ""
1512 0 ""
1612 0 ""
0113 0 ""
0213 0 ""
0313 0 "woods:1;foliage_elev:2"
0413 1 "woods:2;foliage_elev:2"
0513 1 "woods:1;foliage_elev:2"
0613 2 ""
0713 2 ""
0813 1 ""
0913 1 "woods:1;foliage_elev:2"
1013 0 "woods:1;foliage_elev:2"
1113 0 ""
1213 0 ""
1313 0 ""
1413 0 ""
1513 0 ""
1613 0 ""
0114 0 ""
0214 0 ""
0314 0 "woods:1;foliage_elev:2"
0414 1 "woods:1;foliage_elev:2"
0514 2 "woods:1;foliage_elev:2"
0614 1 ""
0714 1 ""
0814 0 ""
0914 0 ""
1014 0 ""
1114 0 ""
1214 0 ""
1314 0 ""
1414 0 ""
1514 0 ""
1614 0 ""
0115 0 ""
0215 0 ""
0315 0 ""
0415 0 ""
0515 1 ""
0615 0 ""
0715 0 ""
0815 0 ""
0915 0 ""
1015 0 ""
1115 0 ""
1215 0 "woods:1;foliage_elev:2"
1315 0 ""
1415 0 "woods:1;foliage_elev:2"
1515 0 ""
1615 0 ""
0116 0 ""
0216 0 ""
0316 0 ""
0416 0 ""
0516 0 ""
0616 0 ""
0716 0 ""
0816 0 ""
0916 0 ""
1016 0 "woods:1;foliage_elev:2"
1116 0 "woods:1;foliage_elev:2"
1216 0 "woods:1;foliage_elev:2"
1316 0 "woods:2;foliage_elev:2"
1416 0 ""
1516 0 ""
1616 0 ""
0117 0 ""
0217 0 ""
0317 0 ""
0417 0 ""
0517 0 ""
0617 0 ""
0717 0 ""
0817 0 ""
0917 0 ""
1017 0 ""
1117 0 ""
1217 0 ""
1317 0 ""
1417 0 ""
1517 0 ""
1617 0 ""
"""


# Module-level singleton — board is static
BOARD = Board()
