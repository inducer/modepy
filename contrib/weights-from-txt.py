#! /usr/bin/env python
from __future__ import annotations

import sys


with open(sys.argv[1]) as inf:
    lines = [ln.strip() for ln in inf if ln.strip()]

rule_name = sys.argv[2]

table = {}

i = 0
while i < len(lines):
    ln = lines[i]
    i += 1
    order, point_count = (int(x) for x in ln.split())

    points = []
    weights = []

    for _ in range(point_count):
        ln = lines[i]
        i += 1
        data = [float(x) for x in ln.split()]
        points.append(data[:-1])
        weights.append(data[-1])

    table[order] = {
            "points": points,
            "weights": weights}

from pprint import pformat


print(f"{rule_name} = {pformat(table)}")

print(f"""

{rule_name} = dict(
    (order, dict((name, numpy.array(ary)) for name, ary in rule.iteritems()))
    for order, rule in {rule_name}.iteritems())
""")
