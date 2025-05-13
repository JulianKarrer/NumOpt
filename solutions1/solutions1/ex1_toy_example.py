import casadi as ca

# create empty optimization problem
opti = ca.Opti()

# define variables
x = opti.variable()      # scalar by default. This is the same as opti.variable(1,1).
y = opti.variable(1,1)   # explicitly define as 1x1 matrix, i.e. scalar

# define objective
f = x**2 - 2*x + y + y**2

# hand objective to casadi, no minimization done yet
opti.minimize(f)    # opti.minimize(x**2 - 2*x + y + y**2) would also work

# define constraints. To include several constraints, just call this
# function several times
opti.subject_to(   x >= 1.5)
opti.subject_to( x+y >= 0  )

# define solver
opti.solver('ipopt')                    # Use IPOPT as solver
# opti.solver('sqpmethod')              # Use sqpmethod as solver

# solve optimization problem
sol = opti.solve()

# read and print solution
xopt = sol.value(x)
yopt = sol.value(y)

print()
if opti.stats()['return_status'] == 'Solve_Succeeded':  # if casadi returns successfully
    print(f"Optimal solution found: x =  {xopt}, y = {yopt}")
else:
    print("Problem failed")
