# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/205451
from __future__ import annotations

import os

import numpy as np


class File:
    def __init__(self, fnam="out.pov", *items):
        self.file = open(fnam, "w")  # noqa: SIM115
        self.__indent = 0
        self.write(*items)

    def include(self, name):
        self.writeln(f'#include "{name}"')
        self.writeln()

    def indent(self):
        self.__indent += 1

    def dedent(self):
        self.__indent -= 1
        assert self.__indent >= 0

    def block_begin(self):
        self.writeln("{")
        self.indent()

    def block_end(self):
        self.dedent()
        self.writeln("}")
        if self.__indent == 0:
            # blank line if this is a top level end
            self.writeln()

    def write(self, *items):
        for item in items:
            if type(item) is str:
                self.include(item)
            else:
                item.write(self)

    def writeln(self, s=""):
        # print "  "*self.__indent+s
        self.file.write("  " * self.__indent + s + os.linesep)


class Vector:
    def __init__(self, *args):
        if len(args) == 1:
            self.v = args[0]
        else:
            self.v = args

    def __str__(self):
        return "<{}>".format(", ".join([str(x) for x in self.v]))

    def __repr__(self):
        return f"Vector({self.v})"

    def __mul__(self, other):
        return Vector([r * other for r in self.v])

    def __rmul__(self, other):
        return Vector([r * other for r in self.v])


class Item:
    def __init__(self, name, args=(), opts=(), **kwargs):
        self.name = name

        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], tuple | list | np.ndarray):
                args[i] = Vector(args[i])

        self.args = args
        self.opts = list(opts)
        self.kwargs = kwargs

    def append(self, item):
        self.opts.append(item)

    def write(self, file):
        file.writeln(self.name)
        file.block_begin()
        if self.args:
            file.writeln(", ".join([str(arg) for arg in self.args]))
        for opt in self.opts:
            if hasattr(opt, "write"):
                opt.write(file)
            else:
                file.writeln(str(opt))
        for key, val in list(self.kwargs.items()):
            if val is None:
                file.writeln(key)
            elif isinstance(val, tuple | list | np.ndarray):
                val = Vector(*val)
                file.writeln(f"{key} {val}")
            else:
                file.writeln(f"{key} {val}")
        file.block_end()

    def __setattr__(self, name, val):
        self.__dict__[name] = val
        if name not in ["kwargs", "args", "opts", "name"]:
            self.__dict__["kwargs"][name] = val

    def __setitem__(self, i, val):
        if i < len(self.args):
            self.args[i] = val
        else:
            i += len(self.args)
            if i < len(self.opts):
                self.opts[i] = val

    def __getitem__(self, i, val):
        if i < len(self.args):
            return self.args[i]
        else:
            i += len(self.args)
            if i < len(self.opts):
                return self.opts[i]


class Texture(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "texture", (), opts, **kwargs)


class Pigment(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "pigment", (), opts, **kwargs)


class Finish(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "finish", (), opts, **kwargs)


class Normal(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "normal", (), opts, **kwargs)


class Camera(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "camera", (), opts, **kwargs)


class Background(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "background", (), opts, **kwargs)


class LightSource(Item):
    def __init__(self, v, *opts, **kwargs):
        Item.__init__(self, "light_source", (Vector(v),), opts, **kwargs)


class Box(Item):
    def __init__(self, v1, v2, *opts, **kwargs):
        # self.v1 = Vector(v1)
        # self.v2 = Vector(v2)
        Item.__init__(self, "box", (v1, v2), opts, **kwargs)


class Cylinder(Item):
    def __init__(self, v1, v2, r, *opts, **kwargs):
        "opts: open"
        Item.__init__(self, "cylinder", (v1, v2, r), opts, **kwargs)


class Plane(Item):
    def __init__(self, v, r, *opts, **kwargs):
        Item.__init__(self, "plane", (v, r), opts, **kwargs)


class Torus(Item):
    def __init__(self, r1, r2, *opts, **kwargs):
        Item.__init__(self, "torus", (r1, r2), opts, **kwargs)


class Cone(Item):
    def __init__(self, v1, r1, v2, r2, *opts, **kwargs):
        "opts: open"
        Item.__init__(self, "cone", (v1, r1, v2, r2), opts, **kwargs)


class Sphere(Item):
    def __init__(self, v, r, *opts, **kwargs):
        Item.__init__(self, "sphere", (v, r), opts, **kwargs)


class Union(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "union", (), opts, **kwargs)


class Intersection(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "intersection", (), opts, **kwargs)


class Difference(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "difference", (), opts, **kwargs)


class Merge(Item):
    def __init__(self, *opts, **kwargs):
        Item.__init__(self, "merge", (), opts, **kwargs)


x = Vector(1, 0, 0)
y = Vector(0, 1, 0)
z = Vector(0, 0, 1)
white = Texture(Pigment(color=(1, 1, 1)))


def tutorial31():
    "from the povray tutorial sec. 3.1"
    file = File("demo.pov", "colors.inc", "stones.inc")
    cam = Camera(location=(0, 2, -3), look_at=(0, 1, 2))
    sphere = Sphere((0, 1, 2), 2, Texture(Pigment(color="Yellow")))
    light = LightSource((2, 4, -3), color="White")
    file.write(cam, sphere, light)


def spiral():
    "Fibonacci spiral"
    gamma = (np.sqrt(5) - 1) / 2
    file = File()
    Camera(location=(0, 0, -128), look_at=(0, 0, 0)).write(file)
    LightSource((100, 100, -100), color=(1, 1, 1)).write(file)
    LightSource((150, 150, -100), color=(0, 0, 0.3)).write(file)
    LightSource((-150, 150, -100), color=(0, 0.3, 0)).write(file)
    LightSource((150, -150, -100), color=(0.3, 0, 0)).write(file)
    theta = 0.0
    for i in range(200):
        r = i * 0.5
        color = 1, 1, 1
        v = [r * np.sin(theta), r * np.cos(theta), 0]
        Sphere(
            v,
            0.7 * np.sqrt(i),
            Texture(
                Finish(ambient=0.0, diffuse=0.0, reflection=0.85, specular=1),
                Pigment(color=color),
            ),
        ).write(file)
        theta += gamma * 2 * np.pi
