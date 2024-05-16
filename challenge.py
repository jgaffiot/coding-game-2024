"""CodingGames challenge: Poker Chip Race."""

import sys
from math import atan2, cos, pi, sin, sqrt
from typing import Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, erfc
from scipy.stats import multivariate_normal

X_MAX = 800
Y_MAX = 515

MY_ID = int(input())  # 0 to 4 (?)
OIL_ID = -1


def debug(string: str) -> None:
    """Print to stderr to debug and avoid conflict with instruction printing."""
    print(string, file=sys.stderr, flush=True)


class Vec2:
    """A 2 dimension vector."""

    def __init__(self, x: float, y: float):
        """Initialize self with cartesian coordinates."""
        self.x = float(x)
        self.y = float(y)

    @property
    def r(self) -> float:
        """Modulus."""
        return sqrt(self.x * self.x + self.y * self.y)

    @property
    def theta(self) -> float:
        """Argument."""
        return atan2(self.y, self.x)

    @classmethod
    def from_polar(cls, r: float, theta: float) -> "Vec2":
        """Create a Vec2 from polar coordinates."""
        return cls(r * cos(theta), r * sin(theta))

    @property
    def np(self) -> np.ndarray:
        """Return the vector as a numpy array."""
        return np.array([self.x, self.y])

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"Vec2({self.x}, {self.y})"

    def __str__(self) -> str:
        """Return str(self)."""
        return f"({self.x}, {self.y})"

    def __add__(self, other: "Vec2") -> "Vec2":
        """Return self + other."""
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        """Return self - other."""
        return Vec2(self.x - other.x, self.y - other.y)

    def __eq__(self, other: "Vec2") -> bool:
        """Return self == other."""
        return (
            self.x == other.x
            and self.y == other.y
            and self.__class__ == other.__class__
        )

    def __mul__(self, scalar: float) -> "Vec2":
        """Return self * other."""
        return Vec2(self.x * scalar, self.y * scalar)

    def __div__(self, scalar: float) -> "Vec2":
        """Return self / other."""
        return Vec2(self.x / scalar, self.y / scalar)


class Point(Vec2):
    """A 2 dimension point."""

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"Point({self.x}, {self.y})"

    def __str__(self) -> str:
        """Return str(self)."""
        return f"P({self.x}, {self.y})"

    def dist_to(self, p: "Point") -> float:
        """Distance to another point."""
        return sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)

    def advance(self, dist: int) -> None:
        """Advance the point from the given dist (same argument)."""
        theta = self.theta
        self.x += dist * cos(theta)
        self.y += dist * sin(theta)
        self.x = min(max(0, self.x), X_MAX)
        self.y = min(max(0, self.y), Y_MAX)

    def __add__(self, other: "Speed") -> "Point":
        """Return self + other."""
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Speed") -> "Point":
        """Return self - other."""
        return Vec2(self.x - other.x, self.y - other.y)


class Speed(Vec2):
    """A 2 dimension speed."""

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"Point({self.x}, {self.y})"

    def __str__(self) -> str:
        """Return str(self)."""
        return f"P({self.x}, {self.y})"


def distance(a: Point, b: Point) -> float:
    """Return the distance between 2 Points."""
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class Chip:
    """A Chip on the play field."""

    def __init__(self):
        """Initialize self."""
        self.id: int = -1
        self.player = -2
        self.radius = 0.0
        self.p: Point = Point(-1, -1)
        self.v: Speed = Speed(0, 0)
        # self.targets: Dict[int, Chip] = {}
        # self.opponents: Dict[int, Chip] = {}

    @classmethod
    def from_string(cls, string: str) -> "Chip":
        """Return a new Chip fully initialized from the given string."""
        new = cls()
        pars = zip(
            ("id_", "player", "r", "x", "y", "vx", "vy"), string.split(), strict=False
        )
        new.update(
            **{
                k: int(v) if k in ("id_", "player") else float(v)
                for k, v in pars.items()
            }
        )
        return new

    @classmethod
    def from_data(
        cls, id_: int, player: int, r: float, x: float, y: float, vx: float, vy: float
    ) -> "Chip":
        """Return a new Chip fully initialized with the given data."""
        new = cls()
        new.update(id_, player, r, x, y, vx, vy)
        return new

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"Chip({self.id}, {self.p.x}, {self.p.y}, {self.pv}, {self.nb_buster})"

    def __str__(self) -> str:
        """Return str(self)."""
        return f"Chip {self.id}, {self.p}, pv={self.pv}, nb={self.nb_buster}"

    @property
    def area(self) -> float:
        """The chip area."""
        return pi * self.r * self.r

    @property
    def future(self) -> Point:
        """Return the future point."""
        return self.p + self.v

    def dist_to(self, p: Point) -> float:
        """Return the distance to a given Point."""
        return self.p.dist_to(p)

    def update(
        self, id_: int, player: int, r: float, x: float, y: float, vx: float, vy: float
    ) -> None:
        """Update the current entity with new data."""
        self.id = id_
        self.player = player
        self.radius = r
        self.p = Point(x, y)
        self.v = Speed(vx, vy)

    def compute_dist_to_chip(self, chips: dict[int, "Chip"]) -> None:
        """Compute the distance to the given chips."""
        self.targets = {id_: Chip(chip, self.p) for id_, chip in chips.items()}

    def get_closest_target(self) -> Union[None, "Chip"]:
        """Get the closest target if any."""
        targets = [t for t in self.targets.values() if t.pv >= 0]
        if targets:
            return sorted(targets, key=lambda t: t.d)[0]
        return None

    def potential(
        self,
        x: np.array,
        y: np.array,
        other_radius: float,
        sig: float = 10,
        alpha: float = 100,
    ) -> np.array:
        """Compute the chip potential."""
        if other_radius == self.r:
            return 0
        sign = 1 if self.r > other_radius else -1
        g0 = multivariate_normal(self.p.np, sig * self.r)
        g_ = multivariate_normal(self.future.np, sig * self.r)
        z = np.dstack((x, y))
        return sign * alpha * g0.pdf(z) + g_.pdf(z)

    def __contains__(self, point: Point) -> bool:
        """Return True if the point is inside the chip or will be at the next point."""
        return self.dist_to(point) < self.r or self.future.dist_to(point) < self.r


class Mine(Chip):
    """One of my chips."""


class Opponent(Chip):
    """An opponent's chip."""


class Oil(Chip):
    """An oil chip."""


def field_potential(x: np.array, y: np.array, sig: float = 10) -> np.array:
    """Compute the field potential."""
    return (
        2.0
        + erfc(x / sig)
        + erfc(y / sig)
        + erf((x - X_MAX) / sig)
        + erf((y - Y_MAX) / sig)
    )


def game_loop():
    """The game loop."""
    chip_registry: dict[int, Chip] = {}
    mine_registry: dict[int, Mine] = {}
    opponent_registry: dict[int, Opponent] = {}
    oil_registry: dict[int, Oil] = {}

    turn_index = 0
    while True:
        turn_index += 1

        # Load the visible entities
        player_chip_count = int(input())  # The number of chips under your control
        # The total number of entities on the table, including your chips
        entity_count = int(input())

        for _ in range(entity_count):
            inputs = input().split()
            chip = Chip.from_string(inputs)

            chip_registry[chip.id_] = chip
            if chip.player == MY_ID:
                mine_registry[chip.id_] = chip
            elif chip.player == OIL_ID:
                oil_registry[chip.id_] = chip
            else:
                opponent_registry[chip.id_] = chip

        total = sum(c.area for c in chip_registry.values())

        # My chip's actions
        for id_, chip in mine_registry.items():
            chip: Mine

            # Fat enough
            if chip.area >= total / 2.0:
                print("WAIT")
                continue

            # Compute the potential and minimize it
            def fun(x_: np.array, *args, id_=id_, chip=chip):
                x = x_[0]
                y = x_[1]
                potential = field_potential(x, y)
                for other_id, other_chip in chip_registry.items():
                    if other_id == id_:
                        continue
                    potential += other_chip.potential(x, y, chip.radius)
                return potential

            res = minimize(fun, chip.np, bounds=((0, X_MAX), (0, Y_MAX)))
            dest = Point(res.x[0], res.x[1])
            debug(dest)

            if dest in chip:
                print("WAIT")
            else:
                print(f"{res.x[0]} {res.x[1]}")


if __name__ == "__main__":
    game_loop()
