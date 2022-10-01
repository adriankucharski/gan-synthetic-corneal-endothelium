"""
Hexagon Mosaic Generator
@author: Adrian Kucharski
"""

import math
from typing import Tuple
import numpy as np
import cv2
import multiprocessing
import itertools


def closest(points: np.ndarray, xy: np.ndarray, k=1) -> np.ndarray:
    # distance = np.linalg.norm(points-xy, axis=1)
    # distance = np.sqrt(np.sum((points - xy) ** 2, axis=1))
    # distance = np.sum((points - xy) ** 2, axis=1)
    distance = np.sum(np.abs(points - xy), axis=1)
    return np.argpartition(distance, k)[:k]


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T).astype("int")


def grid_create_hexagons(
    hex_size: float = 30,
    neatness: float = 0.7,
    width: int = 64,
    height: int = 64,
    random_shift: int = None,
    seed: int = None,
    remove_edges_ratio=0.0,
    rotation=None,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed=seed)
    hex_l1 = hex_size / 2.0
    hex_l2 = hex_l1 * 2.0 / math.sqrt(3.0)
    hex_l3 = hex_l1 / math.sqrt(3.0)
    hex_width = 6 * hex_l1 / math.sqrt(3.0)
    hex_height = hex_size
    rows = int((height + hex_height - 1) / hex_height)
    cols = int((width + hex_width * 2 - 1) / hex_width)
    rand_localize = int(hex_size * (1.0 - neatness))

    points = []
    for i in range(-1, rows + 1, 1):
        for j in range(-1, cols + 1, 1):
            x1 = hex_width * j + hex_l3
            y1 = hex_height * i
            x2 = x1 + hex_l2
            y2 = y1
            x3 = x2 + hex_l3
            y3 = y2 + hex_l1
            x4 = x1 - hex_l3
            y4 = y1 + hex_l1
            points.extend([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    points = np.array(points, dtype="int")
    nearest = []
    for i in range(len(points)):
        nearest.append(closest(points, points[i], 4))

    if rand_localize > 0:
        points += np.random.randint(
            -rand_localize // 2, rand_localize // 2, (len(points), 2)
        )

    if random_shift is not None and random_shift > 0:
        points += np.random.randint(0, random_shift)

    grid = np.zeros((width, height, 1))

    remove_p = np.random.choice(
        [True, False], p=(remove_edges_ratio, 1 - remove_edges_ratio), size=len(nearest)
    )

    if rotation is not None and rotation != 0:
        points = rotate(points, (width // 2, height // 2), rotation)

    for i, ips in enumerate(nearest):
        x, y = points[i]
        if x >= 0 and y >= 0 and x < width and y < height and remove_p[i] == False:
            for nxy in points[ips]:
                cv2.line(grid, points[i], nxy, 1, 1)
    return grid


def generate_hexagons(
    num: int,
    hex_size: Tuple[int, int] = (16, 20),
    neatness: float = 0.7,
    width: int = 64,
    height: int = 64,
    random_shift: int = 6,
    remove_edges_ratio=0.0,
    rotation_range=(0, 0),
) -> np.ndarray:
    smin, smax = hex_size
    random_hex_size = np.random.uniform(smin, smax, num)

    if rotation_range is None:
        rotation_range = (0, 0)
    rotation_range = np.random.uniform(*rotation_range, num)

    args = zip(
        random_hex_size,
        itertools.repeat(neatness),
        itertools.repeat(width),
        itertools.repeat(height),
        itertools.repeat(random_shift),
        itertools.repeat(None),
        itertools.repeat(remove_edges_ratio),
        rotation_range,
    )
    # multi-thread approach
    pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
    hexagons = pool.starmap(grid_create_hexagons, args)
    return np.array(hexagons)
