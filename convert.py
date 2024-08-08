import warnings

warnings.filterwarnings("ignore")

import argparse
import numpy as np
import gpxpy
import gpxpy.gpx
import xml.etree.ElementTree as ET
import numpy.polynomial as poly
from scenariogeneration import xodr
from lxml import etree
from utils import (
    angle_between,
    compute_slopes,
    find_hard_turns,
    find_roundabouts_from_gpx,
    generate_segments,
    geometry_from_points,
    latlon_to_xy,
    xy_to_uv,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", type=str, help="input gpx file path", required=True
)
parser.add_argument(
    "-o", "--output", type=str, help="output xodr file path", default="o.xodr"
)
parser.add_argument("--lanewidth", type=float, help="lane width", default=6.0)
parser.add_argument(
    "--turnth", type=float, help="turn angle threshold in degree", default=135
)
args = parser.parse_args()

with open(args.input, "r") as f:
    gpx = gpxpy.parse(f)

# Extract track points
gpx_points = []
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            gpx_points.append(point)

# sanity check data removing roundabouts points
roundabout_mask = find_roundabouts_from_gpx(
    gpx_points, turn_angle_threshold=args.turnth
)
# gpx_points = [p for idx, p in enumerate(gpx_points) if not roundabout_mask[idx]]

latitudes = np.array([p.latitude for p in gpx_points])
longitudes = np.array([p.longitude for p in gpx_points])

# conversion of latitude, longitude in x, y coordinates
x, y = latlon_to_xy(latitudes, longitudes)
e = np.array([p.elevation for p in gpx_points])

# find hard turns points
turn_angle_threshold = args.turnth
hard_turns = find_hard_turns(x, y, turn_angle_threshold)
hard_turn_idxs = np.where(hard_turns)[0]

# walking algorithm to build slices considering hard turns
segments = generate_segments(len(x), hard_turn_idxs, seg_length=3)

# generate geometry from slices
lengths = np.array(
    [p_t.distance_2d(p_tp1) for p_t, p_tp1 in zip(gpx_points[:-1], gpx_points[1:])]
)

# calculate first heading direction
heading_x, heading_y = np.diff(x)[0], np.diff(y)[0]

# build roads
geometries = []
for s in segments:
    start, end = s[0], s[-1]
    segment_x, segment_y = x[start : end + 1], y[start : end + 1]
    ls = lengths[start:end]
    p = np.pad(ls, (1, 0)).cumsum()

    theta = np.arctan2(heading_y, heading_x)  # heading angle in radians
    geometry, (poly_u, poly_v, err) = geometry_from_points(
        segment_x, segment_y, p, theta
    )

    # retrieve next heading vector
    heading_u, heading_v = poly_u.deriv()(p[-1]), poly_v.deriv()(p[-1])
    heading_x, heading_y = xy_to_uv(heading_u, heading_v, 0, 0, -theta)
    geometries += [geometry]

road = xodr.create_road(
    geometries, 1, left_lanes=1, right_lanes=1, lane_width=args.lanewidth
)

# generate elevation profile
cum_lengths = np.cumsum(np.pad(lengths, (1, 0)))

slopes = np.array(compute_slopes(cum_lengths, e))
slope_th = 0.025
hard_slopes = np.zeros_like(e, dtype=bool)
hard_slopes[:-1] = slopes > slope_th
hard_slopes_idxs = np.where(hard_slopes)[0]

elevation_segments = generate_segments(len(e), hard_turn_idxs, seg_length=4)

# add elevation profile to road
for s in elevation_segments:
    start, end = s[0], s[-1]
    segment_e = e[start : end + 1]
    ls = lengths[start:end]
    p = np.pad(ls, (1, 0)).cumsum()
    poly_e = poly.Polynomial(poly.polynomial.polyfit(p, segment_e, 3))
    road.add_elevation(cum_lengths[start], *poly_e.coef)

# build xodr file
odr = xodr.OpenDrive("myroad")
odr.add_road(road)

odr.adjust_roads_and_lanes()
odr.adjust_elevations()
odr.adjust_roadmarks()

parser = etree.XMLParser(remove_blank_text=True)
lxml_element = etree.fromstring(ET.tostring(odr.get_element(), "utf-8"), parser=parser)
xodr_xml = etree.tostring(lxml_element)
with open(args.output, "wb") as f:
    f.write(xodr_xml)
