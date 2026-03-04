# FGO translated to Python
# Runs on a 2d simple function and visualizes the hyphae trajectory

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# ==============================
# Initialization Function
# ==============================
def initialization(SearchAgents_no, dim, ub, lb):

    ub = np.array(ub)
    lb = np.array(lb)

    Boundary_no = len(ub)

    # If all variables share same bound
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb

    # If each dimension has different bounds
    else:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Positions


# ==============================
# Fungal Growth Optimizer (FGO)
# ==============================
def FGO(N, Tmax, ub, lb, dim, fhd):

    # ---------------- Definitions ----------------
    Gb_Sol = np.zeros(dim)
    Gb_Fit = np.inf
    Conv_curve = np.zeros(Tmax)

    # ---------------- Controlling Parameters ----------------
    M = 0.6
    Ep = 0.7
    R = 0.9

    # ---------------- Initialization ----------------
    S = initialization(N, dim, ub, lb)
    Sp = np.copy(S)

    fit = np.zeros(N)
    t = 0

    # ---------------- Initial Evaluation ----------------
    for i in range(N):
        fit[i] = fhd(S[i, :])

    idx = np.argmin(fit)
    Gb_Fit = fit[idx]
    Gb_Sol = S[idx, :].copy()


    # ================= Optimization Loop =================
    history = []    # store S
    while t < Tmax:

        # -------- Nutrient allocation --------
        if t <= Tmax / 2:
            nutrients = np.random.rand(N)
        else:
            nutrients = fit.copy()

        nutrients = nutrients / (np.sum(nutrients) + 1e-12) + 2 * np.random.rand()

        # ==================================================
        if np.random.rand() < np.random.rand():
            # -------- Hyphal tip growth --------
            for i in range(N):

                # Random distinct indices
                while True:
                    a = np.random.randint(0, N)
                    b = np.random.randint(0, N)
                    c = np.random.randint(0, N)
                    if len(set([i, a, b, c])) == 4:
                        break

                p = (fit[i] - np.min(fit)) / (np.max(fit) - np.min(fit) + 1e-12)
                Er = M + (1 - t / Tmax) * (1 - M)

                if p < Er:
                    # Differential-style growth
                    F = (fit[i] / (np.sum(fit) + 1e-12)) * np.random.rand() * \
                        (1 - t / Tmax) ** (1 - t / Tmax)
                    E = np.exp(F)

                    r1 = np.random.rand(dim)
                    r2 = np.random.rand()
                    U1 = r1 < r2

                    S[i, :] = U1 * S[i, :] + \
                               (~U1) * (S[i, :] + E * (S[a, :] - S[b, :]))

                else:
                    Ec = (np.random.rand(dim) - 0.5) * np.random.rand() * (S[a, :] - S[b, :])

                    if np.random.rand() < np.random.rand():
                        # Opposite direction
                        De2 = np.random.rand(dim) * (S[i, :] - Gb_Sol) * \
                              (np.random.rand(dim) > np.random.rand())
                        S[i, :] = S[i, :] + De2 * nutrients[i] + Ec * (np.random.rand() > np.random.rand())
                    else:
                        # Toward nutrient-rich area
                        De = np.random.rand() * (S[a, :] - S[i, :]) + \
                             np.random.rand(dim) * \
                             ((np.random.rand() > np.random.rand() * 2 - 1) * Gb_Sol - S[i, :]) * \
                             (np.random.rand() > R)

                        S[i, :] = S[i, :] + De * nutrients[i] + Ec * (np.random.rand() > Ep)

                # -------- Boundary Handling --------
                for j in range(dim):
                    if S[i, j] > ub[j]:
                        S[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                    elif S[i, j] < lb[j]:
                        S[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])

                # -------- Fitness Evaluation --------
                nF = fhd(S[i, :])

                # -------- Greedy Selection --------
                if fit[i] < nF:
                    S[i, :] = Sp[i, :].copy()
                else:
                    Sp[i, :] = S[i, :].copy()
                    fit[i] = nF

                    if fit[i] <= Gb_Fit:
                        Gb_Sol = S[i, :].copy()
                        Gb_Fit = fit[i]

                t += 1
                if t >= Tmax:
                    break

                Conv_curve[t - 1] = Gb_Fit

            if t >= Tmax:
                break

        # ==================================================
        else:
            # -------- Branching / Spore --------
            r5 = np.random.rand()

            for i in range(N):

                while True:
                    a = np.random.randint(0, N)
                    b = np.random.randint(0, N)
                    c = np.random.randint(0, N)
                    if len(set([i, a, b, c])) == 4:
                        break

                if np.random.rand() < 0.5:
                    # Branching
                    EL = 1 + np.exp(fit[i] / (np.sum(fit) + 1e-12)) * \
                         (np.random.rand() > np.random.rand())

                    Dep1 = S[b, :] - S[c, :]
                    Dep2 = S[a, :] - Gb_Sol

                    r1 = np.random.rand(dim)
                    r2 = np.random.rand()
                    U1 = r1 < r2

                    S[i, :] = S[i, :] * U1 + \
                               (S[i, :] + r5 * Dep1 * EL + (1 - r5) * Dep2 * EL) * (~U1)

                else:
                    # Spore germination
                    sig = 1 if np.random.rand() > np.random.rand() * 2 - 1 else -1

                    F = (fit[i] / (np.sum(fit) + 1e-12)) * np.random.rand() * \
                        (1 - t / Tmax) ** (1 - t / Tmax)
                    E = np.exp(F)

                    for j in range(dim):
                        mu = sig * np.random.rand() * E
                        if np.random.rand() > np.random.rand():
                            S[i, j] = (
                                ((t / Tmax) * Gb_Sol[j] +
                                 (1 - t / Tmax) * S[a, j]) +
                                S[b, j]
                            ) / 2.0 + \
                                       mu * abs(
                                (S[c, j] + S[a, j] + S[b, j]) / 3.0 - S[i, j]
                            )

                # -------- Boundary Handling --------
                for j in range(dim):
                    if S[i, j] > ub[j]:
                        S[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                    elif S[i, j] < lb[j]:
                        S[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])

                # -------- Fitness --------
                nF = fhd(S[i, :])

                if fit[i] < nF:
                    S[i, :] = Sp[i, :].copy()
                else:
                    Sp[i, :] = S[i, :].copy()
                    fit[i] = nF

                    if fit[i] <= Gb_Fit:
                        Gb_Sol = S[i, :].copy()
                        Gb_Fit = fit[i]

                t += 1
                if t >= Tmax:
                    break

                Conv_curve[t - 1] = Gb_Fit

            if t >= Tmax:
                break
        history.append(S.copy())

    return Gb_Fit, Gb_Sol, Conv_curve, history




def simple_2d(x):
    return (x[0] - 3)**2 + (x[1] + 2)**2


# -------- parameters ----------
dim = 2
lb = [-10, -10]
ub = [10, 10]
N = 10
Tmax = 500

# --------- run -----------
Gb_Fit, Gb_Sol, Conv_curve, history = FGO(
    N=N,
    Tmax=Tmax,
    ub=ub,
    lb=lb,
    dim=dim,
    fhd=simple_2d
)

print("Best Fitness:", Gb_Fit)
print("Best Solution:", Gb_Sol)



# Convert list to array
history = np.array(history)  # shape: (T, N, 2)

# Plot contour of objective
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
Z = (X - 3)**2 + (Y + 2)**2

plt.contour(X, Y, Z, levels=30)

# Plot each agent trajectory
for i in range(history.shape[1]):
    plt.plot(history[:, i, 0], history[:, i, 1], alpha=0.6)
    plt.scatter(history[:, i, 0], history[:, i, 1], alpha=0.6, s=20)

# Plot final best
plt.scatter(Gb_Sol[0], Gb_Sol[1], color='red', s=100)

plt.title("Hyphae Trajectories")
plt.savefig("fgo_trajectory.png")
plt.close()
