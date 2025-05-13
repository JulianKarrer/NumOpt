# L2 Fitting
# %%
import matplotlib.pyplot as plt
import numpy as np

## Question 3.
rng = np.random.default_rng(0)  # random number generator
N = 30
x = np.linspace(0,5,N)
y_no_noise = 3*x + 4                   # true output
y = y_no_noise + rng.normal(size=N)          # add noise

# visualize
fig, ax = plt.subplots()
ax.set_title("No outliers")
ax.plot(x, y_no_noise, 'r', label=r"True output")
ax.plot(x, y, 'ko', label=r'noisy measurements')
ax.legend()

## Question 4.
J = np.column_stack((x, np.ones(N)))
ab = np.linalg.inv(J.T @ J) @ (J.T @ y)  # solve linear system
a = ab[0]  # slope
b = ab[1]  # intercept
ax.plot(x, a*x + b, 'b', label=r'fitted line with L2 norm')
ax.legend()
# %%
## Question 5.
# introduce outliers
y2      = np.copy(y)
y2[0]   = 20
y2[4]   = 17
y2[-1]  = 10

fig, ax = plt.subplots()
ax.set_title("With outliers")
ax.plot(x, y_no_noise, 'r', label=r'True output')
ax.plot(x, y2, 'ko', label=r'noisy measurements')

ab_L2 = np.linalg.inv(J.T @ J) @ (J.T @ y2)  # solve linear system
a_L2, b_L2 = ab_L2[0], ab_L2[1]
ax.plot(x, a_L2*x + b_L2, 'b', label=r'fitted line with L2 norm')
ax.legend()


# %%
# L1 Fitting
## Question 2.
# %%
import casadi as ca


opti = ca.Opti() # Create an optimization problem


# Create variables
a = opti.variable(1)  # slope
b = opti.variable(1)  # intercept

s = opti.variable(N)  # slacks


# Create the cost function
cost = ca.sum1(s)  # cost function
opti.minimize(cost)


# Create the constraints
residuals = a * x + b - y
opti.subject_to( s >= residuals )
opti.subject_to( s >= - residuals )

# Solve the problem
opti.solver('ipopt')  # solve the optimization problem
sol = opti.solve()  # solve the optimization problem


# Retrieve the solution
a_L1 = sol.value(a)  # slope
b_L1 = sol.value(b)  # intercept

# %%
# Plot the results
fig, ax = plt.subplots()
ax.plot(x, y_no_noise, 'r', label=r'measurements no noise')
ax.plot(x, y2, 'ko', label=r'noisy measurements')

ax.plot(x, a_L1*x + b_L1, 'g', label=r'fitted line with L1 norm')
ax.plot(x, a_L2*x + b_L2, 'b', label=r'fitted line with L2 norm')
ax.legend()
# %%
