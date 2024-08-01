import numpy as np
from scenariogeneration import xodr
from numpy.polynomial import polynomial as P


def latlon_to_xy(lat, lon):
    R = 6378137  # Radius of the Earth in meters
    x = R * np.deg2rad(lon) * np.cos(np.deg2rad(lat))
    y = R * np.deg2rad(lat)  # np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2))
    return x, y


def rotate_coordinates(x, y, theta):  # theta in rad
    _x = x * np.cos(theta) + y * np.sin(theta)
    _y = y * np.cos(theta) - x * np.sin(theta)
    return _x, _y


def translate_coordinates(x, y, p, q):  # translation by vector(p, q)
    _x = x + p
    _y = y + q
    return _x, _y


def xy_to_uv(x, y, p, q, theta):  # p, q origin coordinates in global coordinates
    u = x * np.cos(theta) + y * np.sin(theta) - p * np.cos(theta) - q * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta) + p * np.sin(theta) - q * np.cos(theta)
    return u, v


def get_parampoly(x, y, p, theta, order=3):
    u, v = xy_to_uv(x, y, x[0], y[0], theta)

    poly_u = P.Polynomial(P.polyfit(p, u, order))
    poly_v = P.Polynomial(P.polyfit(p, v, order))
    return poly_u, poly_v, u, v


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def geometry_from_points(segment_x, segment_y, p, theta, order=3):
    poly_u, poly_v, u, v = get_parampoly(segment_x, segment_y, p, theta, order=order)
    coef_u = poly_u.coef
    if len(coef_u) < 4:
        coefs = np.zeros(4)
        coefs[: len(coef_u)] = coef_u
        coef_u = coefs
    coef_v = poly_v.coef
    if len(coef_v) < 4:
        coefs = np.zeros(4)
        coefs[: len(coef_v)] = coef_v
        coef_v = coefs
    geometry = xodr.ParamPoly3(*coef_u, *coef_v, prange="arcLength", length=p[-1])

    reconstructed_u, reconstructed_v = poly_u(p), poly_v(p)
    err = (
        (np.array([reconstructed_u, reconstructed_v]) - np.array([u, v])) ** 2
    ).mean()

    return geometry, (poly_u, poly_v, err)


def generate_segments(n_points, repulsive_idxs, seg_length=3):
    i = 0
    segments = []
    current_segment = []

    assert seg_length > 2
    while i < n_points:
        current_segment += [i + k for k in range(seg_length - 1)]
        i += seg_length - 2
        if ((i + 1) in repulsive_idxs) and (current_segment[-1] not in repulsive_idxs):
            segments += [current_segment]
            current_segment = []
        else:
            current_segment += [i + 1]
            i += 1
            while len(current_segment) > 0 and (current_segment[-1] in repulsive_idxs):
                current_segment += [i]
                i += 1
            segments += [current_segment]
            current_segment = []
    segments[-1] = [i for i in segments[-1] if i < n_points]
    if len(segments[-1]) < 2:
        segments = segments[:-1]
    return segments


def compute_slopes(p, e):
    slopes = []
    for i in range(len(e) - 1):
        x1, y1 = p[i], e[i]
        x2, y2 = p[i + 1], e[i + 1]
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
    return slopes
