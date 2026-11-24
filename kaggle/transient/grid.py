import json

import numpy as np

WS = 2 * np.pi * 50.0
GEN = [0, 1, 2]
PMU_BUS = [3, 6, 8]
NET = 9


def branches(raw):
    g = json.load(open(raw / "system_info" / "grid_model.json"))
    return {b["name"].split()[1]: (b["from_bus"] - 1, b["to_bus"] - 1, b["R_pu"], b["X_pu"], b["B_pu"])
            for b in g["branches"]}


def flow_angle(P, Vi, Vj, R, X):
    y = 1 / (R + 1j * X)
    c = (Vi ** 2 * y.real - P) / (Vi * Vj)
    a = np.arccos(np.clip(c / abs(y), -1, 1))
    phi = np.arctan2(y.imag, y.real)
    c1 = (phi - a + np.pi) % (2 * np.pi) - np.pi
    c2 = (phi + a + np.pi) % (2 * np.pi) - np.pi
    return np.where(np.abs(c1) < np.abs(c2), c1, c2)


def x57(br, cap_on):
    return np.where(cap_on, 0.55, 1.0) * br["L5-7"][3]


def measured_cap(br, scada, th47_pmu, cap_logged):
    V = {k: scada[f"V{k}"].values for k in (4, 5, 7)}
    t45 = flow_angle(scada["P_L4-5"].values, V[4], V[5], br["L4-5"][2], br["L4-5"][3])
    err = [np.abs(t45 + flow_angle(scada["P_L5-7"].values, V[5], V[7], br["L5-7"][2], x57(br, on)) - th47_pmu)
           for on in (False, True)]
    known = np.isfinite(err[0]) & np.isfinite(err[1])
    return np.where(known, err[1] < err[0], cap_logged)


def prefault_voltages(br, scada, cap_on):
    V = np.stack([scada[f"V{k}"].values for k in range(1, 10)], 1)
    th = np.zeros_like(V)
    P = lambda c: scada[c].values
    fa = lambda p, i, j, name, x=None: flow_angle(p, V[:, i], V[:, j], br[name][2],
                                                  br[name][3] if x is None else x)
    th[:, 4] = th[:, 3] - fa(P("P_L4-5"), 3, 4, "L4-5")
    th[:, 6] = th[:, 4] - fa(P("P_L5-7"), 4, 6, "L5-7", x57(br, cap_on))
    th[:, 5] = th[:, 3] - fa(P("P_L4-6"), 3, 5, "L4-6")
    th[:, 8] = th[:, 5] - fa(P("P_L6-9"), 5, 8, "L6-9")
    th[:, 7] = th[:, 6] - fa(P("P_L7-8"), 6, 7, "L7-8")
    for g, (name, bus, pcol) in enumerate([("T1", 3, "P_G1"), ("T2", 6, "P_G2"), ("T3", 8, "P_G3")]):
        th[:, g] = th[:, bus] + np.arcsin(np.clip(P(pcol) * br[name][3] / (V[:, g] * V[:, bus]), -1, 1))
    return V * np.exp(1j * th)


def network(br, n, cap_on, l56_on):
    Y = np.zeros((n, NET, NET), complex)
    for name, (i, j, R, X, B) in br.items():
        x = x57(br, cap_on) if name == "L5-7" else np.full(n, X)
        on = l56_on if name == "L5-6" else np.ones(n, bool)
        y = on / (R + 1j * x)
        sh = on * 1j * B / 2
        Y[:, i, i] += y + sh
        Y[:, j, j] += y + sh
        Y[:, i, j] -= y
        Y[:, j, i] -= y
    return Y


def augment(Ynet, yload, xd, fault=None, br=None, cap_on=None):
    n = len(Ynet)
    m = NET + 3 + (fault is not None)
    Y = np.zeros((n, m, m), complex)
    Y[:, :NET, :NET] = Ynet
    Y[:, np.arange(NET), np.arange(NET)] += yload
    for g in range(3):
        y = 1 / (1j * xd[:, g])
        a, b = GEN[g], NET + g
        Y[:, a, a] += y
        Y[:, b, b] += y
        Y[:, a, b] -= y
        Y[:, b, a] -= y
    if fault is not None:
        line, x, rf = fault
        i, j, R, X, _ = br[line]
        xl = x57(br, cap_on) if line == "L5-7" else np.full(n, X)
        y = 1 / (R + 1j * xl)
        f = m - 1
        Y[:, i, i] -= y
        Y[:, j, j] -= y
        Y[:, i, j] += y
        Y[:, j, i] += y
        for k, w in ((i, x), (j, 1 - x)):
            yy = y / w
            Y[:, k, k] += yy
            Y[:, f, f] += yy
            Y[:, k, f] -= yy
            Y[:, f, k] -= yy
        Y[:, f, f] += 1 / np.maximum(rf, 1e-4)
    return Y


def kron(Y):
    gen = [NET, NET + 1, NET + 2]
    net = [k for k in range(Y.shape[1]) if k not in gen]
    Ynn = Y[:, net][:, :, net]
    Yng = Y[:, net][:, :, gen]
    Ygn = Y[:, gen][:, :, net]
    Ygg = Y[:, gen][:, :, gen]
    return Ygg - Ygn @ np.linalg.solve(Ynn, Yng), Ynn, Yng


def fault_bus_voltages(Y, E):
    _, Ynn, Yng = kron(Y)
    return -np.linalg.solve(Ynn, (Yng @ E[:, :, None]))[:, :, 0]


def electrical_power(Yred, Emag, delta):
    E = Emag * np.exp(1j * delta)
    return (E * np.conj(np.einsum("nij,nj->ni", Yred, E))).real


def without_line(br, Ynet, line, cap_on):
    i, j, R, X, B = br[line]
    xl = x57(br, cap_on) if line == "L5-7" else np.full(len(Ynet), X)
    y = 1 / (R + 1j * xl)
    Y = Ynet.copy()
    Y[:, i, i] -= y + 1j * B / 2
    Y[:, j, j] -= y + 1j * B / 2
    Y[:, i, j] += y
    Y[:, j, i] += y
    return Y


class Machine:
    def __init__(self, br, scada, line, cap_on, l56_on, xd, H, damping=0.0):
        n = len(scada)
        V = prefault_voltages(br, scada, cap_on)
        self.ynet = network(br, n, cap_on, l56_on)
        S = V * np.conj(np.einsum("nij,nj->ni", self.ynet, V))
        self.yload = np.zeros((n, NET), complex)
        for k in range(3, NET):
            self.yload[:, k] = -np.conj(S[:, k]) / np.abs(V[:, k]) ** 2
        E = V[:, GEN] + 1j * xd * np.conj(S[:, GEN] / V[:, GEN])
        self.br, self.line, self.cap_on, self.xd = br, line, cap_on, xd
        self.E, self.Emag, self.delta0 = E, np.abs(E), np.angle(E)
        self.M = 2 * H / WS
        self.D = damping / WS
        self.pre = kron(augment(self.ynet, self.yload, xd))[0]
        self.Pm = electrical_power(self.pre, self.Emag, self.delta0)
        self.post = kron(augment(without_line(br, self.ynet, line, cap_on), self.yload, xd))[0]

    def fault(self, x, rf):
        return augment(self.ynet, self.yload, self.xd, (self.line, x, rf), self.br, self.cap_on)

    def step(self, Yred, delta, omega, dt):
        omega = omega + dt * (self.Pm - electrical_power(Yred, self.Emag, delta) - self.D * omega) / self.M
        return delta + dt * omega, omega

    def unstable(self, delta, omega, schedule, dt, horizon):
        out = np.zeros(len(delta), bool)
        for duration, Yred in schedule + [(horizon, self.post)]:
            for _ in range(int(round(duration / dt))):
                delta, omega = self.step(Yred, delta, omega, dt)
                spread = delta.max(1) - delta.min(1)
                out |= spread > np.pi
        return out

    def cct(self, Yfault, reclose=None, tmax=0.8, iters=8, dt=0.005, horizon=1.5):
        Yf = kron(Yfault)[0]
        steps = int(round(tmax / dt))
        n = len(Yf)
        traj = np.zeros((steps + 1, n, 3))
        speed = np.zeros((steps + 1, n, 3))
        traj[0], omega = self.delta0, np.zeros((n, 3))
        for k in range(steps):
            traj[k + 1], omega = self.step(Yf, traj[k], omega, dt)
            speed[k + 1] = omega
        schedule = [] if reclose is None else [(reclose[0], self.post), (reclose[1], Yf)]
        lo, hi = np.zeros(n), np.full(n, tmax)
        never = ~self.unstable(traj[-1], speed[-1], schedule, dt, horizon)
        for _ in range(iters):
            mid = (lo + hi) / 2
            k = np.round(mid / dt).astype(int)
            bad = self.unstable(traj[k, np.arange(n)], speed[k, np.arange(n)], schedule, dt, horizon)
            hi = np.where(bad, mid, hi)
            lo = np.where(bad, lo, mid)
        return np.where(never, tmax, (lo + hi) / 2)
