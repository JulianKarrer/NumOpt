import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


with open("minAction_approx.save") as f:
    data = json.load(f)

xs = lambda i: [x[1] for x in data["xs"][i]]
ay = lambda i: data["ay"][i]
N = len(data["ay"])

fig, ax = plt.subplots()
scat = ax.scatter([], [])
ax.set_xlim(min(min(xs(i)) for i in range(N)), max(max(xs(i)) for i in range(N)))
ax.set_ylim(min(min(ay(i)) for i in range(N)), max(max(ay(i)) for i in range(N)))
ax.set_xlabel("x")
ax.set_ylabel("ay")

def update(i):
    x = xs(i)
    y = ay(i)
    scat.set_offsets(list(zip(x, y)))
    ax.set_title(f"i = {i}")
    return scat,

is_indices = range(N)
ani = FuncAnimation(fig, update, frames=is_indices, blit=True, repeat=False)

plt.show()
