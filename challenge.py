"""
CodingGames challenge: Poker Chip Race.

To read a debug map:
>>> import plotly.graph_objects as go
>>> X_MAX = 800
>>> Y_MAX = 515
>>> X, Y = np.meshgrid(np.linspace(0, X_MAX, 1000), np.linspace(0, Y_MAX, 1000))
>>> Z = np.fromfile("map.bin").reshape((1000,1000))
>>> go.Figure(data=[go.Surface(x=X, y=Y, z=Z)]).show()
"""

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from math import atan2, cos, isclose, pi, sin, sqrt, tan
from typing import Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, erfc
from scipy.stats import multivariate_normal

VERBOSE = False

X_MAX = 800
Y_MAX = 515
BOUNDS = np.array(((0, X_MAX), (0, Y_MAX)))

OIL_ID = -1

Potential = Callable[[np.ndarray], float]
Jacobian = Callable[[np.ndarray], np.ndarray]


def debug(stuff: object) -> None:
    """Print to stderr to debug and avoid conflict with instruction printing."""
    print(stuff, file=sys.stderr, flush=True)


def get_input() -> str:
    """Get an input."""
    val = input()
    if VERBOSE:
        debug(val)
    return val


class Dist(ABC):
    """A distribution and its jacobian."""

    @abstractmethod
    def pdf(self, x_: np.ndarray) -> float:
        """Return the probability density function."""
        raise NotImplementedError

    @abstractmethod
    def jac(self, x_: np.ndarray) -> np.ndarray:
        """Return the Jacobian."""
        raise NotImplementedError


class NormalDist(Dist):
    """The normal multivariate distribution."""

    def __init__(self, x0: np.ndarray, sig: float, alpha: float = 1.0) -> None:
        """Initialize self."""
        self.x0 = x0
        self.sig = sig
        self.alpha = alpha
        self.dist = multivariate_normal(x0, sig)

    def pdf(self, x_: np.ndarray) -> float:
        """Return the probability density function."""
        return self.alpha * self.dist.pdf(x_)

    def jac(self, x_: np.ndarray) -> np.ndarray:
        """Return the Jacobian."""
        return self.alpha * self.dist.pdf(x_) * (x_ - self.x0) / self.sig


class ErfDist(Dist):
    """The erf 2-dim distribution."""

    def __init__(self, x0: np.ndarray, sig: float) -> None:
        """Initialize self."""
        self.x0 = x0
        self.sig = sig
        self.cst = 2 / np.sqrt(np.pi) / sig

    def pdf(self, x_: np.ndarray) -> float:
        """Return the probability density function."""
        return np.max(erf((x_ - self.x0) / self.sig), axis=len(x_.shape) - 1)

    def jac(self, x_: np.ndarray) -> np.ndarray:
        """Return the Jacobian."""
        return self.cst * np.exp(-(((x_ - self.x0) / self.sig) ** 2))


class ErfcDist(Dist):
    """The erfc = 1 - erf 2-dim distribution."""

    def __init__(self, x0: np.ndarray, sig: float) -> None:
        """Initialize self."""
        self.x0 = x0
        self.sig = sig
        self.cst = 2 / np.sqrt(np.pi) / sig

    def pdf(self, x_: np.ndarray) -> float:
        """Return the probability density function."""
        return np.max(erfc((x_ - self.x0) / self.sig), axis=len(x_.shape) - 1)

    def jac(self, x_: np.ndarray) -> np.ndarray:
        """Return the Jacobian."""
        return -self.cst * np.exp(-(((x_ - self.x0) / self.sig) ** 2))


class Vec2:
    """A 2 dimension vector."""

    def __init__(self, x: float, y: float):
        """Initialize self with cartesian coordinates."""
        self.x = float(x)
        self.y = float(y)

    @property
    def rho(self) -> float:
        """Modulus."""
        return sqrt(self.x * self.x + self.y * self.y)

    @property
    def theta(self) -> float:
        """Argument."""
        return atan2(self.y, self.x)

    @classmethod
    def from_polar(cls, rho: float, theta: float) -> "Vec2":
        """Create a Vec2 from polar coordinates."""
        return cls(rho * cos(theta), rho * sin(theta))

    @property
    def np(self) -> np.ndarray:
        """Return the vector as a numpy array."""
        return np.array([self.x, self.y])

    def unit(self) -> "Vec2":
        """Return the unit vector co-linear with self."""
        return self / self.rho

    def rot(self, theta: float) -> "Vec2":
        """Return a vector rotated by the given angle in radian."""
        return Vec2.from_polar(self.rho, (self.theta + theta) % (2 * pi))

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

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, Vec2):
            return NotImplemented

        return (
            self.x == other.x
            and self.y == other.y
            and self.__class__ == other.__class__
        )

    def __mul__(self, scalar: float) -> "Vec2":
        """Return self * scalar."""
        return Vec2(self.x * scalar, self.y * scalar)

    def __matmul__(self, other: "Vec2") -> float:
        """Return self * other."""
        return self.x * other.x + self.y * other.y

    def __truediv__(self, scalar: float) -> "Vec2":
        """Return self / other."""
        return Vec2(self.x / scalar, self.y / scalar)

    def __copy__(self) -> "Vec2":
        """Return copy(self)."""
        return self.copy()

    def copy(self) -> "Vec2":
        """Return self.copy()."""
        return Vec2(self.x, self.y)


class Point(Vec2):
    """A 2 dimension point."""

    @classmethod
    def from_polar(cls, rho: float, theta: float) -> "Point":
        """Create a Vec2 from polar coordinates."""
        return cls(rho * cos(theta), rho * sin(theta))

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"Point({self.x:.4f}, {self.y:.4f})"

    def __str__(self) -> str:
        """Return str(self)."""
        return f"P({self.x:.2f}, {self.y:.2f})"

    @property
    def out(self) -> str:
        """Return the format expected by the challenge."""
        return f"{self.x} {self.y}"

    def dist_to(self, p: "Point") -> float:
        """Distance to another point."""
        return sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)

    def forward(self, dist: int) -> None:
        """Advance the point from the given dist (same argument)."""
        self.x += dist * cos(self.theta)
        self.y += dist * sin(self.theta)
        self.x = min(max(0.0, self.x), X_MAX)
        self.y = min(max(0.0, self.y), Y_MAX)

    def __add__(self, other: "Speed") -> "Point":  # type: ignore[override]
        """Return self + other."""
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Union["Speed", "Point"]) -> "Point":  # type: ignore[override]
        """Return self - other."""
        return Point(self.x - other.x, self.y - other.y)


class Speed(Vec2):
    """A 2 dimension speed."""

    @classmethod
    def from_polar(cls, rho: float, theta: float) -> "Speed":
        """Create a Vec2 from polar coordinates."""
        return cls(rho * cos(theta), rho * sin(theta))

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"Speed({self.x}, {self.y})"

    def __str__(self) -> str:
        """Return str(self)."""
        return f"V({self.x}, {self.y})"


def distance(a: Point, b: Point) -> float:
    """Return the distance between 2 Points."""
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class Chip:
    """A Chip on the play field."""

    def __init__(self) -> None:
        """Initialize self."""
        self.id: int = -1
        self.player = -2
        self.r = 0.0
        self.p: Point = Point(-1, -1)
        self.v: Speed = Speed(0, 0)
        # self.targets: dict[int, Chip] = {}
        # self.opponents: dict[int, Chip] = {}

    @classmethod
    def from_string(cls, string: str) -> "Chip":
        """Return a new Chip fully initialized from the given string."""
        new = cls()
        pars = zip(
            ("id_", "player", "r", "x", "y", "vx", "vy"), string.split(), strict=False
        )
        new.update(
            **{k: int(v) if k in ("id_", "player") else float(v) for k, v in pars}  # type: ignore[arg-type]
        )
        return new

    @classmethod
    def from_data(  # noqa: PLR0913
        cls, id_: int, player: int, r: float, x: float, y: float, vx: float, vy: float
    ) -> "Chip":
        """Return a new Chip fully initialized with the given data."""
        new = cls()
        new.update(id_, player, r, x, y, vx, vy)
        return new

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"Chip({self.id}, {self.player}, {self.r}, {self.p}, {self.v})"

    def __str__(self) -> str:
        """Return str(self)."""
        return f"Chip {self.id}, {self.player}, {self.r}, {self.p}, {self.v}"

    def __contains__(self, point: Point) -> bool:
        """Return True if the point is inside the chip or will be at the next point."""
        return self.dist_to(point) < self.r or self.future.dist_to(point) < self.r

    @property
    def area(self) -> float:
        """The chip area."""
        return pi * self.r * self.r

    @property
    def future(self) -> "Chip":
        """Return the future chip."""
        bounds = BOUNDS + np.array(((self.r, -self.r), (self.r, -self.r)))

        p = self.p + self.v
        v = self.v.copy()

        if p.x < bounds[0][0]:
            p.x = 2 * bounds[0][0] - p.x
            v.x = -v.x
        elif p.x > bounds[0][1]:
            p.x = 2 * bounds[0][1] - p.x
            v.x = -v.x

        if p.y < bounds[1][0]:
            p.y = 2 * bounds[1][0] - p.y
            v.y = -v.y
        elif p.y > bounds[1][1]:
            p.y = 2 * bounds[1][1] - p.y
            v.y = -v.y

        return Chip.from_data(self.id, self.player, self.r, p.x, p.y, v.x, v.y)

    def dist_to(self, p: Point) -> float:
        """Return the distance to a given Point."""
        return self.p.dist_to(p)

    def update(  # noqa: PLR0913
        self, id_: int, player: int, r: float, x: float, y: float, vx: float, vy: float
    ) -> None:
        """Update the current entity with new data."""
        self.id = id_
        self.player = player
        self.r = r
        self.p = Point(x, y)
        self.v = Speed(vx, vy)

    # def compute_dist_to_chip(self, chips: dict[int, "Chip"]) -> None:
    #     """Compute the distance to the given chips."""
    #     self.targets = {id_: self.p.dist_to(chip) for id_, chip in chips.items()}

    # def get_closest_target(self) -> Union[None, "Chip"]:
    #     """Get the closest target if any."""
    #     targets = [t for t in self.targets.values() if t.pv >= 0]
    #     if targets:
    #         return sorted(targets, key=lambda t: t.d)[0]
    #     return None

    def make_potential(
        self,
        other_radius: float,
        sig: float = 10,
        alpha: float = 10,
    ) -> tuple[Potential, Jacobian]:
        """Return the PDF and the Jacobian of the chip potential."""
        if other_radius == self.r:
            return lambda x_: np.zeros(x_.shape[:-1]), lambda x_: np.array(  # type: ignore[return-value]
                [np.zeros(x_.shape[:-1]), np.zeros(x_.shape[:-1])]
            )

        if self.r < other_radius:
            sign = -1.0 * alpha * self.r
            g1 = NormalDist(self.future.p.np, sig * sig * self.r)
            g2 = NormalDist(self.future.future.p.np, sig * self.r)
        else:
            sign = max(alpha, self.r) ** 2
            g1 = NormalDist(self.p.np, other_radius + self.r)
            g2 = NormalDist(self.future.p.np, other_radius + self.r)

        def pdf(x_: np.ndarray) -> float:
            """Return the chip potential."""
            return sign * (g1.pdf(x_) + g2.pdf(x_))

        def jac(x_: np.ndarray) -> np.ndarray:
            """Return the chip Jacobian."""
            return sign * (g1.jac(x_) + g2.jac(x_))

        return pdf, jac


class Mine(Chip):
    """One of my chips."""


class Opponent(Chip):
    """An opponent's chip."""


class Oil(Chip):
    """An oil chip."""


def make_field_potential(sig: float = 10) -> tuple[Potential, Jacobian]:
    """Return the PDF and the Jacobian of the field potential."""
    bottom_left = ErfcDist(np.array([0, 0]), sig)
    top_right = ErfDist(np.array([X_MAX, Y_MAX]), sig)

    def pdf(x_: np.ndarray) -> float:
        """Return the field potential."""
        return bottom_left.pdf(x_) + top_right.pdf(x_)

    def jac(x_: np.ndarray) -> np.ndarray:
        """Return the field Jacobian."""
        return bottom_left.jac(x_) + top_right.jac(x_)

    return pdf, jac


def compute_order(dest: Point, chip: Chip) -> Point:
    """Compute the order allowing to go to the destination."""
    rho = 200.0 / 14.0
    path = dest - chip.p
    tan_ = tan(path.theta)

    if (
        rho * tan_ - chip.v.x * tan_ + chip.v.y == 0.0
        or rho**2 * (1 + tan_**2) < (chip.v.x * tan_ - chip.v.y) ** 2
    ):
        theta = pi
    else:
        theta = 2 * (
            atan2(
                rho * tan_ - chip.v.x * tan_ + chip.v.y,
                -sqrt(rho**2 * (1 + tan_**2) - (chip.v.x * tan_ - chip.v.y) ** 2) - rho,
            )
        )
    dv: Speed = Speed.from_polar(rho, theta)
    return chip.p + dv


def game_loop() -> None:  # noqa: C901,PLR0912,PLR0915
    """Run the game loop."""
    my_id = int(get_input())  # 0 to 4 (?)

    field_pot = make_field_potential(sig=10)

    turn_index = 0
    while True:
        turn_index += 1

        chip_registry: dict[int, Chip] = {}
        mine_registry: dict[int, Chip] = {}
        opponent_registry: dict[int, Chip] = {}
        oil_registry: dict[int, Chip] = {}

        # Load the visible entities
        # The number of chips under your control
        player_chip_count = int(get_input())  # noqa: F841
        # The total number of entities on the table, including your chips
        entity_count = int(get_input())

        for _ in range(entity_count):
            inputs = get_input()
            chip: Chip = Chip.from_string(inputs)

            chip_registry[chip.id] = chip
            if chip.player == my_id:
                mine_registry[chip.id] = chip
            elif chip.player == OIL_ID:
                oil_registry[chip.id] = chip
            else:
                opponent_registry[chip.id] = chip

        total = sum(c.area for c in chip_registry.values())

        # My chip's actions
        for chip_id, chip in mine_registry.items():
            # Fat enough
            if chip.area >= total / 2.0:
                debug("Fat enough!")
                print("WAIT")
                continue

            # Compute the potential and minimize it
            other_chip_pot: tuple[tuple[Potential, Jacobian], ...] = tuple(
                chip_.make_potential(chip.r, sig=100, alpha=1000)
                for id_, chip_ in chip_registry.items()
                if id_ != chip_id
            )

            def pdf(x_: np.ndarray, chip_pots: tuple = other_chip_pot) -> float:
                """Return the field potential."""
                return field_pot[0](x_) + np.sum([p[0](x_) for p in chip_pots], axis=0)

            # def jac(x_: np.ndarray, chip_pots: tuple = other_chip_pot) -> np.ndarray:
            #     """Return the field Jacobian."""
            #     return field_pot[1](x_) + np.sum([p[1](x_) for p in chip_pots],
            #     axis=0)

            # X, Y = np.meshgrid(np.linspace(0, X_MAX, 1000),
            # np.linspace(0, Y_MAX, 1000))
            # Z: np.ndarray = pdf(np.dstack((X, Y)))
            # Z.tofile("map.bin")

            res = minimize(
                pdf,
                chip.p.np,
                bounds=BOUNDS,
                # jac=jac,
                method="Nelder-Mead",
                # options={"disp": True},
            )
            if VERBOSE:
                debug(res)
            dest = Point(res.x[0], res.x[1])
            order = compute_order(dest, chip)
            debug(f"{dest=} => {order=}")
            debug(f"{dest.theta=}/{order.theta=}/{(order+chip.v).theta=})")

            if dest in chip:
                debug(f"dest in chip: {dest} in {chip.p}+-{chip.r}")
                print("WAIT")
            elif isclose(chip.v.rho, 0.0):
                debug("start moving")
                print(order.out)
            else:
                path = dest - chip.p
                if ((chip.v.theta - path.theta + pi / 2) % (2 * pi)) > pi:
                    debug("going backward")
                    print(order.out)
                elif abs(path @ chip.v.rot(pi / 2).unit()) < chip.r:
                    debug(f"going in the good direction: {chip.v.theta/pi*180.}°")
                    print("WAIT")
                else:
                    debug("missing the dest")
                    print(order.out)


if __name__ == "__main__":
    game_loop()
