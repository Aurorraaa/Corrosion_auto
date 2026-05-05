import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

class CorrosionCA:
    EMPTY = 0
    METAL = 1
    FRONT = 2

    def __init__(self, n=120, m=160, p_base=0.01, p_neighbor=0.08, seed=42):
        self.n = n
        self.m = m
        self.p_base = p_base
        self.p_neighbor = p_neighbor
        self.rng = np.random.default_rng(seed)

        self.grid = np.full((n, m), self.METAL, dtype=np.int8)
        self.grid[0, :] = self.EMPTY
        self.resistance = self.rng.uniform(0.7, 1.3, size=(n, m))
        self.damage_history = []
        self.update_front()

    def count_empty_neighbors(self, i, j):
        count = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.n and 0 <= nj < self.m:
                    if self.grid[ni, nj] == self.EMPTY:
                        count += 1
        return count

    def update_front(self):
        self.grid[self.grid == self.FRONT] = self.METAL
        new_front = []
        for i in range(self.n):
            for j in range(self.m):
                if self.grid[i, j] == self.METAL:
                    if self.count_empty_neighbors(i, j) > 0:
                        new_front.append((i, j))

        for i, j in new_front:
            self.grid[i, j] = self.FRONT

    def step(self):
        new_grid = self.grid.copy()

        for i in range(self.n):
            for j in range(self.m):
                if self.grid[i, j] != self.FRONT:
                    continue

                damaged_neighbors = self.count_empty_neighbors(i, j)
                p = (self.p_base + self.p_neighbor * damaged_neighbors) / self.resistance[i, j]
                p = min(p, 1.0)

                if self.rng.random() < p:
                    new_grid[i, j] = self.EMPTY

        self.grid = new_grid
        self.update_front()

        damage_percent = 100 * np.sum(self.grid == self.EMPTY) / (self.n * self.m)
        self.damage_history.append(damage_percent)


def main():
    frames = 300
    model = CorrosionCA(n=120, m=160, p_base=0.01, p_neighbor=0.08, seed=7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Цвета:
    # EMPTY = черный
    # METAL = синий
    # FRONT = оранжевый
    cmap = ListedColormap(["black", "steelblue", "orange"])

    im = ax1.imshow(
        model.grid,
        cmap=cmap,
        vmin=0,
        vmax=2,
        extent=(0, 100, 100, 0),
        animated=True
    )
    ax1.set_title("Клеточный автомат коррозии")
    ax1.set_xlabel("x, %")
    ax1.set_ylabel("y, %")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(100, 0)
    ax1.set_xticks(np.arange(0, 101, 20))
    ax1.set_yticks(np.arange(0, 101, 20))

    line, = ax2.plot([], [], lw=2)
    ax2.set_xlim(0, frames)
    ax2.set_ylim(0, 100)
    ax2.set_yticks(np.arange(0, 101, 20))
    ax2.margins(x=0, y=0)
    ax2.autoscale(False)
    ax2.set_title("Степень разрушения")
    ax2.set_xlabel("Шаг")
    ax2.set_ylabel("% разрушенных клеток")
    ax2.grid(True)

    def init():
        im.set_data(model.grid)
        line.set_data([], [])
        return im, line

    def update(frame):
        model.step()
        im.set_data(model.grid)

        x = np.arange(1, len(model.damage_history) + 1)
        y = model.damage_history
        line.set_data(x, y)

        ax1.set_title(f"Клеточный автомат коррозии — шаг {frame}")
        return im, line

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=50,
        blit=True
    )

    plt.tight_layout()
    anim.save("corrosion.gif", writer="pillow", fps=20)
    plt.show()

if __name__ == "__main__":
    main()
