import casadi as ca

opti = ca.Opti()

w1 = opti.variable()
w2 = opti.variable()
w3 = opti.variable()

f = 2*w1**2 + w1*w3 + 2*w3**2 + 3*w2 - ca.log(w3 + 1)

opti.minimize(f)
opti.subject_to( -2*w1**2 - 0.5*w2**2 + 3 >= 0 )
opti.subject_to( 1 <= w3 )
opti.subject_to( w3 <= 4 )

opti.solver('ipopt')
# opti.set_initial(0,0)

sol = opti.solve()
w1s = sol.value(w1)
w2s = sol.value(w2)
w3s = sol.value(w3)

print(sol, "\nw1s =",w1s, "\nw2s =",w2s, "\nw3s =",w3s)