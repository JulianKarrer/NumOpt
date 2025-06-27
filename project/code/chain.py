# type: ignore
import taichi as ti
import taichi.math as tm 
import numpy as np
import matplotlib.pyplot as plt
import time

ti.init(arch=ti.gpu, debug=True)

N = 10_000
x = ti.Vector.field(n=2, dtype=ti.f32, shape=(N,), needs_grad=True) # positions
D = 5.0                                                             # spring stiffness of each spring
M = 0.1                                                             # mass at each point
v = ti.field(dtype=ti.f32, shape=(), needs_grad=True)               # scalar current energy of the system


def init():
    x.fill(0.)
    x[0] = ti.Vector([0.,0.])
    x[N-1] = ti.Vector([5.,0.])
    v[None] = 0.

@ti.kernel
def compute_energy():
    for i in range(1,N):
        dx = x[i] - x[i-1]
        v[None] += (0.5*ti.static(D)*tm.dot(dx,dx) + 9.81*ti.static(M)*x[i].y)/N

@ti.kernel
def Qx(x:ti.template(), out:ti.template()):
    out[0]=x[0]-x[1]
    out[N-1]=x[N-1]-x[N-2]
    for i in range(1,N-1):
        out[i] = -x[i-1] +2*x[i]  -x[i+1] 

@ti.kernel
def steepest_descent_step():
    for i in range(1,N-1):
        x[i] -= 1. * x.grad[i]

def descent():
    for _ in range(100_000):
        with ti.ad.Tape(loss=v):
            compute_energy()
        steepest_descent_step()

init()
descent()
print(x, v, x.grad)

