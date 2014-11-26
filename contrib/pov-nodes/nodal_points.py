from __future__ import division
from __future__ import absolute_import

from pov import Sphere, Cylinder, File, Union, Texture, Pigment, \
        Camera, LightSource, Plane, Background, Finish
import numpy as np
import modepy as mp
import six
from six.moves import range
from six.moves import zip

n = 8

ball_radius = 0.05
link_radius = 0.02

from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
        as gnitstam
node_tuples = list(gnitstam(n, 3))
faces = [
        [nt for nt in node_tuples if nt[0] == 0],
        [nt for nt in node_tuples if nt[1] == 0],
        [nt for nt in node_tuples if nt[2] == 0],
        [nt for nt in node_tuples if sum(nt) == n]
        ]

from modepy.tools import unit_to_barycentric, barycentric_to_equilateral
nodes = [(n[0],n[2], n[1]) for n in
        barycentric_to_equilateral(
            unit_to_barycentric(
                mp.warp_and_blend_nodes(3, n, node_tuples))).T]
id_to_node = dict(list(zip(node_tuples, nodes)))

def get_ball_radius(nid):
    in_faces = len([f for f in faces if nid in f])
    if in_faces >= 2:
        return ball_radius * 1.333
    else:
        return ball_radius

def get_ball_color(nid):
    in_faces = len([f for f in faces if nid in f])
    if in_faces >= 2:
        return (1,0,1)
    else:
        return (0,0,1)

balls = Union(*[
    Sphere(node, get_ball_radius(nid),
        Texture(Pigment(color=get_ball_color(nid)))
        )
    for nid, node in six.iteritems(id_to_node)
    ])

links = Union()

for nid in node_tuples:
    child_nids = []
    for i in range(len(nid)):
        nid2 = list(nid)
        nid2[i] += 1
        child_nids.append(tuple(nid2))

    def connect_nids(nid1, nid2):
        try:
            links.append(Cylinder(
                id_to_node[nid1],
                id_to_node[nid2],
                link_radius))
        except KeyError:
            pass

    for i, nid2 in enumerate(child_nids):
        connect_nids(nid, nid2)
        connect_nids(nid2, child_nids[(i+1)%len(child_nids)])

links.append(Texture(
    Pigment(color=(0.8,0.8,0.8)),
    Finish(
        specular=1,
        ),
    ))

outf = File("nodes.pov")

Camera(location=0.65*np.array((4,0.8,-1)), look_at=(0,0.1,0)).write(outf)
LightSource(
        (10,5,0),
        color=(1,1,1),
        ).write(outf)
Background(
        color=(1,1,1)
        ).write(outf)
if False:
    Plane(
            (0,1,0), min(n[1] for n in nodes)-ball_radius,
            Texture(Pigment(color=np.ones(3,)))
            ).write(outf)
balls.write(outf)
links.write(outf)
