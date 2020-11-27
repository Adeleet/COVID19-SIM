import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from scipy.spatial import distance_matrix
import matplotlib.patches as mpatches


class Simulation:
    def __init__(self, N=100, INITIAL_INFECTED=5, P_CURE=0.01, DISTANCE_DECAY=-0.6):
        # Initialize population [x, y, isInfected]
        self.population = np.random.rand(N, 3) * N
        self.vector_infected = [INITIAL_INFECTED]
        self.vector_immune = [[]]

        self.N = N
        self.P_CURE = 0.01
        self.DISTANCE_DECAY = -0.6

        # Set initial infected population to 1/N
        self.population[:, 2] = [1] * INITIAL_INFECTED + [0] * (N - INITIAL_INFECTED)

        # Calculate vector of infection probabilities based on distance

        self._create_figure_()

    def _calc_infection_probabilities_(self, distances):
        vfunc = np.vectorize(lambda d: np.math.exp(self.DISTANCE_DECAY * d))
        return vfunc(distances)

    def _run_infection_bin_trails_(self, infection_probabilities):
        # Run N binomial probability trials based on distance-infection probabilities
        vfunc = np.vectorize(lambda probs: np.random.binomial(1, probs))
        return vfunc(infection_probabilities)

    def simulate_movement(self, k=1):
        movements = k * np.random.standard_normal((self.N, 2))
        self.population[:, :2] = np.clip(self.population[:, :2] + movements, 0, self.N)

    def _get_color_(self, i):
        if i in self.vector_immune[-1]:
            return "deepskyblue"
        elif round(self.population[i, 2]) == 1:
            return "tomato"
        else:
            return "limegreen"

    def get_plot_colors_(self):
        return [self._get_color_(i) for i in range(self.N)]

    def _create_figure_(self):
        plt.style.use("seaborn")
        fig, axes = plt.subplots(2, 1, figsize=(7, 7))
        fig.tight_layout()

        fig.suptitle("COVID-19 Epidemic Simulation")
        axes[1].set_title("Infection Statistics over time")
        axes[1].set_xlabel("Days")
        axes[0].set_facecolor("0.95")

        axes[0].grid(False)

        self.fig = fig
        self.axes = axes

    def _clear_figure_(self):
        self.axes[0].clear()
        self.axes[0].set_xlim([-self.N * 0.05, self.N * 1.05])
        self.axes[0].set_ylim([-self.N * 0.05, self.N * 1.05])

    def simulate_cures(self):
        infected = np.where(self.population[:, 2] == 1)[0]
        cured_trials = infected * np.random.binomial(1, self.P_CURE, infected.size)
        cured_indices = cured_trials[cured_trials != 0]

        immune = self.vector_immune[-1].copy()

        for idx in cured_indices:
            if idx not in immune:
                immune.append(idx)

        self.vector_immune.append(immune)

    def simulate_infections(self):
        # Compute distance matrix
        distances = distance_matrix(self.population[:, :2], self.population[:, :2])

        # Calculate infection probabilities
        infection_probabilities = self._calc_infection_probabilities_(distances)

        # Update infections in population
        trials = self._run_infection_bin_trails_(infection_probabilities)
        infections = np.dot(self.population[:, 2], trials)
        self.population[:, 2] = np.clip(infections, 0, 1)

        self.population[self.vector_immune[-1], 2] = 0

        self.vector_infected.append(self.population[:, 2].sum())

    def step(self):
        self._clear_figure_()
        self.simulate_movement()

        self.simulate_cures()

        self.simulate_infections()

    def update_figure(self):

        colors = self.get_plot_colors_()

        # 0 plot positions
        self.axes[0].scatter(self.population[:, 0], self.population[:, 1], c=colors)

        # 1A plot infections
        self.axes[1].plot(self.vector_infected, c="tomato")

        immune_progress = [len(imm_pop) for imm_pop in self.vector_immune]

        susceptible = [
            self.N - self.vector_infected[i] - immune_progress[i]
            for i in range(len(immune_progress))
        ]
        self.axes[1].plot(susceptible, c="limegreen")

        self.axes[1].plot(immune_progress, c="deepskyblue")

    def animate(self, i):
        self.step()
        self.update_figure()


sim = Simulation()

animation = FuncAnimation(sim.fig, func=sim.animate, interval=20)
plt.show()
