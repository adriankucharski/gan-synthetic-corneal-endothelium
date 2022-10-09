"""
Hexagon Mosaic Generator
@author: Adrian Kucharski
"""

from skimage import io
import math
from typing import Tuple, List
from matplotlib import pyplot as plt
import numpy as np
import cv2
import multiprocessing
import itertools


def closest(points: np.ndarray, xy: np.ndarray, k=1) -> np.ndarray:
    distance = np.linalg.norm(points - xy, axis=1)
    # distance = np.sqrt(np.sum((points - xy) ** 2, axis=1))
    # distance = np.sum((points - xy) ** 2, axis=1)
    # distance = np.sum(np.abs(points - xy), axis=1)
    return np.argpartition(distance, k)[:k]


def rotate(p: np.ndarray, origin=(0, 0), degrees=0) -> np.ndarray:
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T).astype("int")


def grid_create_hexagons_mosaic(
    hexagon_height: float = 30,
    neatness: float = 0.7,
    width: int = 64,
    height: int = 64,
    random_shift: int = None,
    seed: int = None,
    remove_edges_ratio=0.0,
    rotation=None,
    side_thickness=1,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed=seed)
    hex_l1 = hexagon_height / 2.0
    hex_l2 = hex_l1 * 2.0 / math.sqrt(3.0)
    hex_l3 = hex_l1 / math.sqrt(3.0)
    hex_width = 6 * hex_l1 / math.sqrt(3.0)
    hex_side = int(hexagon_height / math.sqrt(3.0))
    rows = int((height + hexagon_height - 1) / hexagon_height)
    cols = int((width + hex_width * 2 - 1) / hex_width)
    rand_localize = int(hexagon_height * (1.0 - neatness))

    points: List[Tuple[int, int]] = []
    for i in range(-1, rows + 1):
        for j in range(-1, cols + 1):
            x1 = hex_width * j + hex_l3
            y1 = hexagon_height * i
            x2 = x1 + hex_l2
            y2 = y1
            x3 = x2 + hex_l3
            y3 = y2 + hex_l1
            x4 = x1 - hex_l3
            y4 = y1 + hex_l1

            points.extend([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    points = np.array(points, dtype="int")

    neighbors: List[List[int]] = []
    for i in range(len(points)):
        candidates = closest(points, points[i], 4)
        point_neighbors: List[int] = []
        for neighbor in candidates:
            dist = int(np.linalg.norm(points[neighbor] - points[i]))
            if neighbor != i and hex_side + 1 >= dist:
                point_neighbors.append(neighbor)
        neighbors.append(point_neighbors)

    if rand_localize > 0:
        points += np.random.randint(
            -rand_localize // 2, rand_localize // 2, (len(points), 2)
        )

    if random_shift is not None and random_shift > 0:
        points += np.random.randint(0, random_shift)

    remove_p = np.random.choice(
        [True, False],
        p=(remove_edges_ratio, 1 - remove_edges_ratio),
        size=len(neighbors),
    )

    if rotation is not None and rotation != 0:
        points = rotate(points, (height // 2, width // 2), rotation)

    grid: np.ndarray = np.zeros((height, width, 1))
    for i, ips in enumerate(neighbors):
        x, y = points[i]
        if x >= 0 and y >= 0 and x < width and y < height and remove_p[i] == False:
            for nxy in points[ips]:
                cv2.line(grid, points[i], nxy, 1, side_thickness)
    return grid


def generate_hexagons(
    num: int,
    hexagon_height: Tuple[int, int] = (16, 20),
    neatness: float = 0.7,
    width: int = 64,
    height: int = 64,
    random_shift: int = 6,
    remove_edges_ratio=0.0,
    rotation_range=(0, 0),
    side_thickness=1
) -> np.ndarray:
    smin, smax = hexagon_height
    random_hexagon_height = np.random.uniform(smin, smax, num)

    if rotation_range is None:
        rotation_range = (0, 0)
    rotation_range = np.random.uniform(*rotation_range, num)

    args = zip(
        random_hexagon_height,
        itertools.repeat(neatness),
        itertools.repeat(width),
        itertools.repeat(height),
        itertools.repeat(random_shift),
        itertools.repeat(None),
        itertools.repeat(remove_edges_ratio),
        rotation_range,
        itertools.repeat(side_thickness)
    )
    # multi-thread approach
    pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
    hexagons = pool.starmap(grid_create_hexagons_mosaic, args)
    return np.array(hexagons)


if __name__ == "__main__":
    w = 4**4
    h = 4**4
    for i in [15, 30, 45, 60, 75, 90]:
        a = grid_create_hexagons_mosaic(17, 1, w, h, rotation=i, side_thickness=1)
        plt.imshow(a, "gray")
        print(w, h)
        plt.show()
