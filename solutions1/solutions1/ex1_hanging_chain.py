import casadi as ca
import matplotlib as mpl
import matplotlib.pyplot as plt


# for people who cannot see the plots
if mpl.get_backend() == 'agg':
    mpl.use('WebAgg')
print(f"backend: {mpl.get_backend()}")


# create empty optimization problem
opti = ca.Opti()

N = 40

y = opti.variable(N,1)
z = opti.variable(N,1)

mi = 4/N
Di = (70/40)*N
g0 = 9.81

# defining the objective
Vchain = 0
# gravitational terms
for i in range(N):
   Vchain += g0*mi*z[i]

# potential energy of the springs
for i in range(N-1):
    Vchain += 1/2*Di*((y[i]-y[i+1])**2 + (z[i]-z[i+1])**2)

# passing the objective to opti
opti.minimize(Vchain)

opti.subject_to( y[0] == -2 )
opti.subject_to( z[0] ==  1 )
opti.subject_to( y[N-1] ==  2 )
opti.subject_to( z[N-1] ==  1 )

# Setting solver to ipopt and solving the problem:
opti.solver('ipopt')
# opti.solver('sqpmethod')
sol = opti.solve()

Y = sol.value(y)
Z = sol.value(z)

# plot the result
fig, ax = plt.subplots()
ax.plot(Y, Z, '--or')
ax.plot(-2, 1, 'xg', markersize=10)
ax.plot(2, 1, 'xg', markersize=10)
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$z$')
ax.set_title(r"Optimal solution hanging chain (without extra constraints)")


## adding the ground constraints
opti.subject_to(       z >= 0.5)
opti.subject_to( z-0.1*y >= 0.5) 

# solve
sol2 = opti.solve()

Y2 = sol2.value(y)
Z2 = sol2.value(z)

# plot
fig, ax = plt.subplots()
ax.plot(Y2,Z2,'--or')
ax.plot(-2, 1, 'xg', markersize=10)
ax.plot(2, 1, 'xg', markersize=10)
ax.plot([-2, 2], [0.5, 0.5], ':b')
ax.plot([-2, 2], [0.3, 0.7], ':b')
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$z$')
ax.set_title(r"Optimal solution hanging chain (with linear ground constraints)")


## show plots
plt.show()
