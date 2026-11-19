"""Flat LCDM background and CosmoPower-JAX P(k) whose construction traces under jax.jit.

The other CosmoPower-JAX classes (:mod:`cloelib.cosmology.cosmopower_jax_cosmology`) build
their tables with NumPy and SciPy, so a likelihood that constructs them cannot be compiled or
vectorised over points. Here every quantity that depends on the sampled cosmology is a JAX
expression, so ``jax.jit(jax.vmap(loglike))`` of a likelihood that builds
:class:`TracedBackground` and :class:`TracedNonLinearPerturbations` from its parameters runs
the whole theory as one compiled function. Everything that does not depend on the sampled
cosmology (the networks, the CAMB density tables, the k and z maps, the dark-matter-only
correction) is built once per configuration and cached.

:class:`TracedBackground`: flat LCDM (w0 = -1, wa = 0) with photons, massless neutrinos and
one massive neutrino species of fixed mass. The radiation and massive-neutrino densities
depend only on T_CMB, N_eff and m_nu, so they are taken once from CAMB
(``get_background_densities``); H(z) is then closed form and chi(z) a 64-node Gauss-Legendre
quadrature in ln(1 + z) (on a uniform grid: to its first node, then by the quadratic through
neighbouring nodes over each interval); the CMB source is at z* = 1089. Omega_m includes the massive neutrinos, as CAMB's.

:class:`TracedNonLinearPerturbations`: ln P(k, z) on a fixed table, ``z`` = the requested
redshifts (with z = 5 added when they reach above it) by ``k`` = ``N_K`` wavenumbers uniform in
ln k on ``K_RANGE``:

- the CosmoPower-JAX HMcode-2020 + feedback network (``lcdm-1mass-nonlinear``) on the
  redshifts up to z = 5, its inputs held to the training box; above its top mode P continues
  as ``extend_spectra``'s ``logk2`` ansatz on its wavenumber nodes, and the table is linear in
  k between them (as the other CosmoPower classes' ``RectBivariateSpline(kx=1, ky=1)``);
- ``nonlinear_model="mead2020"`` (dark matter only, which no emulator covers): the feedback
  network at log T_AGN = 7.8 less HMcode-2020's feedback response ln S(k, z)
  (:mod:`cloelib.cosmology.hmcode2020_feedback_jax`), plus ln[P_CAMB / P] tabulated once from
  CAMB's mead2020 at ``correction_reference``; away from that cosmology this approximates CAMB;
- above z = 5: P(k, z) = P(k, 5) [D(z)/D(5)]^2, D the cold-matter linear growth on this
  background (D = a at z = 200, RK4 in ln a).

The growth factor (``growth_factor``, the NLA amplitude's D) is sqrt[P(z, k_1)/P(0, k_1)] at
the network's second mode k_1, linear in P between table redshifts, and D(5) D(z)/D(5) above
z = 5. sigma8 is the ``lcdm-1mass-s8-fs8`` network at z = 0 (log T_AGN the point's under
feedback, 7.6 under mead2020).

:meth:`TracedNonLinearPerturbations.limber_matter_power_spectrum` gathers P at the Limber
wavenumbers of :class:`~cloelib.summary_statistics.angular_two_point.AngularTwoPoint` from ln P
itself: a fixed not-a-knot cubic spline in z onto the Limber redshifts, then four-point
Lagrange in ln k with the interval found arithmetically; beyond the table ln P continues
linearly in ln k with the edge slope. It replaces the 2-D Akima search of ``Pkl_interp``,
which is most of a Limber evaluation's cost.
"""

from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp

from cloelib.auxiliary.units import SPEED_OF_LIGHT

jax.config.update("jax_enable_x64", True)

C_KMS = SPEED_OF_LIGHT / 1000.0
#: Wavenumbers of the P(k) table (Mpc^-1), uniform in ln k.
K_RANGE = (1e-4, 50.0)
N_K = 400
#: The emulator's trained redshift range and parameter box (cloelib CP_EMULATOR_BOUNDS).
EMULATOR_Z_MAX = 5.0
EMULATOR_INPUTS = ["ombh2", "omch2", "H0", "ns", "lnAs", "z", "mnu", "logT_AGN"]
EMULATOR_BOUNDS = {"ombh2": (0.001, 0.1), "omch2": (0.05, 0.9), "H0": (20.0, 100.0),
                   "ns": (0.6, 1.3), "lnAs": (1.61, 5.0), "mnu": (0.0, 1.0),
                   "logT_AGN": (7.3, 8.5)}
NONLINEAR_FILE = "lcdm-1mass-nonlinear.npz"
SIGMA_FILE = "lcdm-1mass-s8-fs8.npz"
LINEAR_FILES = ("lcdm-1mass-linear.npz", "lcdm-1mass-cb-linear.npz")
#: ``extend_spectra``'s default wavenumber nodes (Mpc^-1).
EXTEND_K = (1e-4, 500.0, 500)
#: The log T_AGN of the dark-matter-only model's emulator rows, and of sigma8 under it.
MEAD2020_LOG_T_AGN = 7.8
LINEAR_LOG_T_AGN = 7.6
GROWTH_Z_START = 200.0
N_GROWTH = 400
#: The feedback response's k nodes (uniform in ln k from RESPONSE_K_MIN to the emulator's top
#: mode), halo masses, and the top of its linear-spectrum grid (Mpc^-1).
RESPONSE_K_MIN = 0.01
RESPONSE_N_K = 32
RESPONSE_N_MASS = 48
RESPONSE_K_LINEAR_MAX = 1e4
#: The CMB source redshift of the CMB-lensing kernel.
Z_STAR = 1089.0
#: Gauss-Legendre nodes of chi(z) in ln(1 + z).
N_CHI = 64
#: The massive-neutrino density table: uniform in ln a on [LN_A_MIN, 0].
LN_A_MIN = -9.0
N_LN_A = 4097


# --------------------------------------------------------------------------- #
# numerics on fixed nodes
# --------------------------------------------------------------------------- #
def spline_matrix(x_nodes, x_eval):
    """Matrix of SciPy's not-a-knot cubic spline through ``x_nodes``, evaluated at ``x_eval``."""
    from scipy.interpolate import CubicSpline
    spline = CubicSpline(np.asarray(x_nodes, float), np.eye(len(x_nodes)), axis=0)
    return np.asarray(spline(np.asarray(x_eval, float)))


def linear_weights(nodes, x):
    """``(i, t)`` with ``x = (1 - t) nodes[i] + t nodes[i+1]``."""
    nodes, x = np.asarray(nodes, float), np.asarray(x, float)
    i = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2)
    return i, (x - nodes[i]) / (nodes[i + 1] - nodes[i])


def lagrange4(table, x):
    """Four-point Lagrange interpolation of ``table`` (..., n) on unit-spaced nodes at
    fractional index ``x`` (..., m), interval held inside the table."""
    n = table.shape[-1]
    i = jnp.clip(jnp.floor(x).astype(int), 1, n - 3)
    t = x - i
    if table.ndim == 1:
        take = lambda j: table[i + j]
    else:
        take = lambda j: jnp.take_along_axis(table, i + j, axis=-1)
    return (-t * (t - 1) * (t - 2) / 6 * take(-1) + (t + 1) * (t - 1) * (t - 2) / 2 * take(0)
            - (t + 1) * t * (t - 2) / 2 * take(1) + (t + 1) * t * (t - 1) / 6 * take(2))


def logk2(lnk, lnp, n_s):
    """``extend_spectra``'s ``logk2`` continuation through the last two nodes ``(lnk, lnp)``:
    P = k^-(4-n_s) A (ln k - ln k*)^2, continuous at the last node. Returns ln P as a function
    of ln k."""
    a = 4.0 - n_s
    k0, k1 = (jnp.exp(a * lnk[j] + lnp[..., j]) for j in (0, 1))
    slope = (k1 - k0) / (lnk[1] - lnk[0])
    lnk_star = lnk[1] - 2.0 * k1 / slope
    ln_amplitude = jnp.log(0.25 * slope ** 2 / k1)
    return lambda lnk_out: (-a * lnk_out + ln_amplitude[..., None]
                            + 2.0 * jnp.log(jnp.abs(lnk_out - lnk_star[..., None])))


def network(net, inputs):
    """A CosmoPower-JAX network's output (log10 P on its modes, or sigma8 and f sigma8), in
    float32 as trained, returned in float64."""
    x = (inputs.astype(jnp.float32) - net["inputs"][0]) / net["inputs"][1]
    for (w, b), (alpha, beta) in zip(net["weights"][:-1], net["hyper"]):
        x = x @ w.T + b
        x = (beta + jax.nn.sigmoid(alpha * x) * (1.0 - beta)) * x
    w, b = net["weights"][-1]
    return ((x @ w.T + b) * net["outputs"][1] + net["outputs"][0]).astype(jnp.float64)


@lru_cache(maxsize=None)
def load_network(filename):
    """A CosmoPower-JAX network as float32 arrays."""
    with jax.ensure_compile_time_eval():
        return _load_network(filename)


def _load_network(filename):
    import warnings
    from cloelib.cosmology.cosmopower_jax_cosmology import emulator_data, load_pk_emulator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = load_pk_emulator(emulator_data(filename))
    f32 = lambda v: jnp.asarray(np.asarray(v, np.float32))
    return dict(parameters=list(emu.parameters),
                weights=[(f32(w), f32(b)) for w, b in emu.weights],
                hyper=[(f32(a), f32(b)) for a, b in emu.hyper_params],
                inputs=(f32(emu.param_train_mean), f32(emu.param_train_std)),
                outputs=(f32(emu.feature_train_mean), f32(emu.feature_train_std)),
                modes=np.asarray(emu.modes, float))


# --------------------------------------------------------------------------- #
# background
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def density_tables(mnu, n_mnu):
    """(omega_r today, a^4 omega_nu(a) on the ln a table) from CAMB, omega in units of
    (100 km/s/Mpc)^2 / (8 pi G / 3): the radiation (photons, massless neutrinos) scales as
    a^-4; the massive neutrinos are tabulated."""
    from cloelib.cosmology.camb_cosmology import CAMBBackground
    background = CAMBBackground(H0=70.0, Omega_b0=0.05, Omega_cdm0=0.25, Omega_k0=0.0, As=2e-9,
                                ns=0.96, mnu=mnu, w0=-1.0, wa=0.0, gamma_MG=0.545, N_mnu=n_mnu)
    a = np.exp(np.linspace(LN_A_MIN, 0.0, N_LN_A))
    d = background.results.get_background_densities(a, vars=["photon", "neutrino", "nu"])
    scale = 3.0 * (100.0 / C_KMS) ** 2
    radiation = np.asarray(d["photon"] + d["neutrino"]) / scale
    return float(radiation[-1]), np.asarray(d["nu"]) / scale


class TracedBackground:
    """Flat LCDM background, traceable (see the module docstring). The constructor takes
    :class:`cloelib.cosmology.cosmology.Background`'s arguments; ``Omega_k0``, ``w0``, ``wa``,
    ``mnu`` and ``N_mnu`` must be concrete, the rest may be traced."""

    def __init__(self, H0, Omega_b0, Omega_cdm0, Omega_k0, As, ns, mnu, w0, wa, gamma_MG,
                 N_mnu, N_ur=None, alpha_s=0.0, **kwargs):
        if Omega_k0 != 0 or w0 != -1 or wa != 0 or alpha_s != 0:
            raise ValueError("TracedBackground is flat LCDM without running")
        if N_mnu not in (0, 1):
            raise ValueError("TracedBackground has at most one massive neutrino species")
        self.H0, self.h = H0, H0 / 100.0
        self.Omega_b0, self.Omega_cdm0, self.Omega_k0 = Omega_b0, Omega_cdm0, 0.0
        self.As, self.ns, self.alpha_s = As, ns, 0.0
        self.mnu, self.N_mnu = float(mnu) if N_mnu else 0.0, int(N_mnu)
        self.w0, self.wa, self.gamma_MG = -1.0, 0.0, gamma_MG
        self._omega_r, nu_table = density_tables(self.mnu, self.N_mnu)
        self._nu_table = jnp.asarray(nu_table)
        h2 = self.h ** 2
        self._omega_cb = (Omega_b0 + Omega_cdm0) * h2
        self._omega_nu0 = float(nu_table[-1])
        self._omega_l = h2 - self._omega_cb - self._omega_r - self._omega_nu0
        self.Omega_m0 = (self._omega_cb + self._omega_nu0) / h2
        self.z_star = Z_STAR
        self._memo = {}

    def _omega_nu(self, a):
        x = (jnp.log(a) - LN_A_MIN) / ((0.0 - LN_A_MIN) / (N_LN_A - 1))
        return lagrange4(self._nu_table, x) / a ** 4

    def dimensionless_hubble(self, a):
        return jnp.sqrt(self._omega_cb / a ** 3 + self._omega_r / a ** 4 + self._omega_nu(a)
                        + self._omega_l) / self.h

    def _once(self, name, zs, compute):
        """``compute(zs)``, kept per (quantity, redshifts) for this background: the tracers and
        every Limber pair ask for the same grids many times in one evaluation."""
        try:
            key = (name, np.shape(zs), np.asarray(zs, float).tobytes())
        except Exception:  # traced redshifts
            return compute(zs)
        if key not in self._memo:
            self._memo[key] = compute(zs)
        return self._memo[key]

    def hubble_parameter(self, zs, units="km/s/Mpc"):
        hz = self._once("H", zs, lambda z: self.H0 * self.dimensionless_hubble(
            1.0 / (1.0 + jnp.asarray(z, float))))
        return hz if units == "km/s/Mpc" else hz / C_KMS

    def comoving_distance(self, zs):
        """chi(z) in Mpc, 64-node Gauss-Legendre in ln(1 + z), for any shape of ``zs``."""
        return self._once("chi", zs, self._comoving_distance)

    def _comoving_distance(self, zs):
        z = np.asarray(zs, float) if not isinstance(zs, jax.core.Tracer) else None
        if z is not None and z.ndim == 1 and len(z) > 2 and z[0] > 0:
            step = np.diff(z)
            if np.allclose(step, step[0], rtol=1e-10, atol=0):
                # A uniform grid: Gauss-Legendre to its first node, then each interval by the
                # quadratic through three neighbouring nodes.
                f = 1.0 / self.dimensionless_hubble(1.0 / (1.0 + z))
                h = step[0]
                first = h / 12.0 * (5.0 * f[:1] + 8.0 * f[1:2] - f[2:3])
                rest = h / 12.0 * (-f[:-2] + 8.0 * f[1:-1] + 5.0 * f[2:])
                cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.concatenate([first, rest]))])
                return self._gauss_legendre_distance(z[:1])[0] + C_KMS / self.H0 * cum
        return self._gauss_legendre_distance(zs)

    def _gauss_legendre_distance(self, zs):
        x, w = np.polynomial.legendre.leggauss(N_CHI)
        top = jnp.log1p(jnp.asarray(zs, float))[..., None]
        u = 0.5 * top * (x + 1.0)
        a = jnp.exp(-u)
        return C_KMS / self.H0 * 0.5 * top[..., 0] * jnp.sum(
            w / (a * self.dimensionless_hubble(a)), axis=-1)

    def Omega_m(self, zs):
        a = 1.0 / (1.0 + jnp.asarray(zs, float))
        om = self._omega_cb / a ** 3 + self._omega_nu(a)
        return om / (self.h * self.dimensionless_hubble(a)) ** 2

    def growth_cb(self, z_query):
        """D_cb(z)/D_cb(5) at ``z_query`` (concrete, each above 5): RK4 in ln a of the cold
        growing mode on this background from D = a at z = 200, Hermite between steps."""
        z_query = np.atleast_1d(np.asarray(z_query, float))
        lna = np.linspace(-np.log1p(GROWTH_Z_START), 0.0, 2 * N_GROWTH + 1)
        a = jnp.asarray(np.exp(lna))
        step = 2 * (lna[1] - lna[0])
        e = self.dimensionless_hubble(a)
        om_cb = self._omega_cb / a ** 3 / (self.h ** 2 * e ** 2)
        coef_d, coef_y = 1.0 / (a ** 2 * e), 1.5 * e * a ** 2 * om_cb

        def body(state, i):
            d, y = state
            k0, k1, k2 = 2 * i, 2 * i + 1, 2 * i + 2
            dd1, dy1 = coef_d[k0] * y, coef_y[k0] * d
            dd2, dy2 = coef_d[k1] * (y + 0.5 * step * dy1), coef_y[k1] * (d + 0.5 * step * dd1)
            dd3, dy3 = coef_d[k1] * (y + 0.5 * step * dy2), coef_y[k1] * (d + 0.5 * step * dd2)
            dd4, dy4 = coef_d[k2] * (y + step * dy3), coef_y[k2] * (d + step * dd3)
            d = d + step / 6 * (dd1 + 2 * dd2 + 2 * dd3 + dd4)
            y = y + step / 6 * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
            return (d, y), (d, y)

        init = (a[0], a[0] ** 3 * e[0])
        _, (ds, ys) = jax.lax.scan(body, init, jnp.arange(N_GROWTH))
        ds = jnp.concatenate([init[0][None], ds])
        slope = jnp.concatenate([init[1][None], ys]) * coef_d[::2]
        coarse = lna[::2]
        nodes = -np.log1p(np.r_[EMULATOR_Z_MAX, z_query])
        i = np.clip(np.searchsorted(coarse, nodes) - 1, 0, len(coarse) - 2)
        t = (nodes - coarse[i]) / step
        h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
        h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        d = h00 * ds[i] + h10 * step * slope[i] + h01 * ds[i + 1] + h11 * step * slope[i + 1]
        return d[1:] / d[0]


class TracedLinearPerturbations:
    """Placeholder linear sector: :class:`TracedNonLinearPerturbations` needs none, and nothing
    of the NLA photometric likelihoods reads it."""

    def __init__(self, background, redshifts, **kwargs):
        self.background, self.z = background, np.asarray(redshifts, float)

    def matter_power_spectrum(self, zs, ks):
        raise NotImplementedError("TracedLinearPerturbations holds no linear P(k)")


# --------------------------------------------------------------------------- #
# P(k)
# --------------------------------------------------------------------------- #
class _Table:
    """What a P(k) table needs that does not depend on the sampled cosmology: the nodes, the
    networks, the maps from the networks' modes to the table and, under mead2020, the feedback
    response and the CAMB correction."""

    def __init__(self, redshifts, mnu, nonlinear_model, reference):
        z = np.asarray(redshifts, float)
        if z[-1] > EMULATOR_Z_MAX and not np.any(z == EMULATOR_Z_MAX):
            z = np.sort(np.r_[z, EMULATOR_Z_MAX])
        self.z, self.mnu = z, mnu
        self.z_emu = z[z <= EMULATOR_Z_MAX]
        self.n_emu = len(self.z_emu)
        self.z_tail = z[z > EMULATOR_Z_MAX]
        self.k = np.exp(np.linspace(np.log(K_RANGE[0]), np.log(K_RANGE[1]), N_K))
        self.feedback = {"mead2020_feedback": True, "mead2020": False}[nonlinear_model]
        self.nonlinear = load_network(NONLINEAR_FILE)
        self.sigma = load_network(SIGMA_FILE)
        for net in (self.nonlinear, self.sigma):
            if net["parameters"] != EMULATOR_INPUTS:
                raise ValueError(f"emulator inputs {net['parameters']} != {EMULATOR_INPUTS}")
        self.k_emu = self.nonlinear["modes"]
        self.k_ia = float(self.k_emu[1])
        # The table's wavenumbers, then the NLA growth's k_1, on the networks' modes continued
        # by logk2 on extend_spectra's nodes.
        query = np.r_[self.k, self.k_ia]
        base = np.geomspace(*EXTEND_K)
        plus = base[base > self.k_emu[-1]]
        self.k_plus = plus[:int(np.searchsorted(plus, query.max())) + 1]
        self.k_i, self.k_t = linear_weights(np.r_[self.k_emu, self.k_plus], query)
        self.ln_correction = np.zeros((len(z), len(query)))
        self.response = None
        if not self.feedback:
            self._build_response(query)
            reference = dict(reference)
            ref = self.reference_cosmology(reference)
            model = np.asarray(self.lnp_table(ref, MEAD2020_LOG_T_AGN))
            camb = self.camb_rows(reference)
            self.ln_correction = np.log(camb) - model

    def reference_cosmology(self, reference):
        h = float(reference["h"])
        return dict(ombh2=reference["Omega_b"] * h ** 2, omch2=reference["Omega_c"] * h ** 2,
                    H0=100.0 * h, ns=float(reference["n_s"]), lnAs=np.log(1e10 * reference["A_s"]),
                    background=TracedBackground(100.0 * h, reference["Omega_b"], reference["Omega_c"],
                                                0.0, reference["A_s"], reference["n_s"], self.mnu,
                                                -1.0, 0.0, 0.545, int(self.mnu > 0)))

    def camb_rows(self, reference):
        """CAMB's mead2020 P on (z, k then k_ia) at the reference cosmology, through cloelib;
        kept in the temporary directory, so the processes of one machine run CAMB once."""
        import hashlib
        import os
        import tempfile
        key = hashlib.sha1(repr((self.z.tolist(), self.k.tolist(), self.k_ia, self.mnu,
                                 sorted(reference.items()))).encode()).hexdigest()
        path = os.path.join(tempfile.gettempdir(), f"cloelib-mead2020-{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        rows = self._camb_rows(reference)
        partial = f"{path}.{os.getpid()}.npy"
        np.save(partial, rows)
        os.replace(partial, path)
        return rows

    def _camb_rows(self, reference):
        from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBNonLinearPerturbations
        background = CAMBBackground(H0=100 * reference["h"], Omega_b0=reference["Omega_b"],
                                    Omega_cdm0=reference["Omega_c"], Omega_k0=0.0,
                                    As=reference["A_s"], ns=reference["n_s"], mnu=self.mnu, w0=-1.0,
                                    wa=0.0, gamma_MG=0.545, N_mnu=int(self.mnu > 0))
        state = CAMBNonLinearPerturbations(background, None, self.z, nonlinear_model="mead2020")
        return np.column_stack([state.matter_power_spectrum(self.z, self.k),
                                state.matter_power_spectrum(self.z, [self.k_ia])])

    def _build_response(self, query):
        from cloelib.cosmology.hmcode2020_feedback_jax import FeedbackResponse
        self.linear = [load_network(f) for f in LINEAR_FILES]
        for net in self.linear:
            if net["parameters"] != EMULATOR_INPUTS[:-1] or not np.array_equal(net["modes"], self.k_emu):
                raise ValueError("the linear networks do not match the nonlinear one")
        lnk = np.log(self.k_emu)
        lnk_linear = np.arange(np.log(1e-5), np.log(RESPONSE_K_LINEAR_MAX), 0.01)
        inside = lnk_linear <= lnk[-1]
        linear_map = np.zeros((len(lnk_linear), len(lnk)))
        linear_map[inside] = spline_matrix(lnk, lnk_linear[inside])
        t = (lnk_linear[~inside] - lnk[-1]) / (lnk[-1] - lnk[-2])
        linear_map[~inside, -1] = 1.0 + t
        linear_map[~inside, -2] = -t
        self.linear_map = linear_map
        lnk_r = np.linspace(np.log(RESPONSE_K_MIN), lnk[-1], RESPONSE_N_K)
        self.response = FeedbackResponse(self.z_emu, np.exp(lnk_r), lnk_linear, n_mass=RESPONSE_N_MASS)
        lnq = np.log(query)
        above = lnq >= lnk_r[0]
        self.response_k = np.zeros((len(query), RESPONSE_N_K))
        self.response_k[above] = spline_matrix(lnk_r, np.minimum(lnq[above], lnk_r[-1]))

    @staticmethod
    def row(cosmo, log_t, mnu):
        """The networks' input row at z = 0, held to the training box."""
        clamp = lambda v, name: jnp.clip(v, *EMULATOR_BOUNDS[name])
        return jnp.stack([clamp(cosmo["ombh2"], "ombh2"), clamp(cosmo["omch2"], "omch2"),
                          clamp(cosmo["H0"], "H0"), clamp(cosmo["ns"], "ns"),
                          clamp(cosmo["lnAs"], "lnAs"), 0.0, mnu, clamp(log_t, "logT_AGN")])

    def lnp_rows(self, cosmo, log_t):
        """ln P of the network on (z_emu, k then k_ia): the network on its modes, logk2 above
        them, linear in k in between."""
        inputs = jnp.tile(self.row(cosmo, log_t, self.mnu), (self.n_emu, 1)).at[:, 5].set(self.z_emu)
        lnp = np.log(10.0) * network(self.nonlinear, inputs)
        plus = logk2(np.log(self.k_emu[-2:]), lnp[:, -2:], cosmo["ns"])(jnp.asarray(np.log(self.k_plus)))
        power = jnp.exp(jnp.concatenate([lnp, plus], axis=1))
        i, t = self.k_i, self.k_t
        return jnp.log((1.0 - t) * power[:, i] + t * power[:, i + 1])

    def ln_response(self, cosmo):
        """ln S of HMcode-2020 at log T_AGN = 7.8 on (z_emu, k then k_ia)."""
        bg = cosmo["background"]
        row = self.row(cosmo, 0.0, self.mnu)[:7]
        tot, cb = (np.log(10.0) * self.linear_map @ network(net, row[None, :])[0] for net in self.linear)
        h = bg.h
        ln_s = self.response.ln_response(tot, cb, cosmo["omch2"] / h ** 2, cosmo["ombh2"] / h ** 2,
                                         bg._omega_nu0 / h ** 2, h, MEAD2020_LOG_T_AGN)
        return ln_s @ self.response_k.T

    def lnp_table(self, cosmo, log_t):
        """ln P on (z, k then k_ia), and D(z)/D(5) on the table's rows above 5."""
        if self.feedback:
            rows = self.lnp_rows(cosmo, log_t)
        else:
            rows = self.lnp_rows(cosmo, MEAD2020_LOG_T_AGN) - self.ln_response(cosmo)
        growth = None
        if len(self.z_tail):
            growth = cosmo["background"].growth_cb(self.z_tail)
            tail = rows[self.n_emu - 1] + 2.0 * jnp.log(growth)[:, None]
            rows = jnp.concatenate([rows, tail])
        return rows + self.ln_correction


@lru_cache(maxsize=None)
def _table(redshifts, mnu, nonlinear_model, reference):
    # Concrete, also when first asked for while a likelihood is being traced.
    with jax.ensure_compile_time_eval():
        return _Table(np.frombuffer(redshifts), mnu, nonlinear_model, reference)


@lru_cache(maxsize=None)
def _limber_maps(z_table, z_limber):
    """The spline from the table's redshifts onto the Limber redshifts (None when equal)."""
    zt, zl = np.frombuffer(z_table), np.frombuffer(z_limber)
    if zt.shape == zl.shape and np.array_equal(zt, zl):
        return None
    with jax.ensure_compile_time_eval():
        return jnp.asarray(spline_matrix(zt, zl))


class TracedNonLinearPerturbations:
    """Nonlinear P(k, z) from CosmoPower-JAX, traceable (see the module docstring).

    Args:
        background: a :class:`TracedBackground`.
        linearperturbations: unused (kept for the interface).
        redshifts: the table's redshifts (concrete).
        log10TAGN: log10 T_AGN/K of the feedback model (default 7.8).
        nonlinear_model: ``"mead2020_feedback"`` (default) or ``"mead2020"``.
        correction_reference: under mead2020, the cosmology (``Omega_c, Omega_b, h, n_s, A_s``)
            at which ln[P_CAMB / P] is tabulated.
        limber_precision: ``"mixed"`` gathers the Limber P in float32 and sets
            :attr:`limber_dtype`, with which cloelib's Limber sums run in float32 too;
            ``"double"`` (default) keeps both in float64.
    """

    limber_dtype = None

    def __init__(self, background, linearperturbations, redshifts, log10TAGN=None,
                 nonlinear_model="mead2020_feedback", correction_reference=None,
                 limber_precision="double", **kwargs):
        self.background = background
        self.linearperturbations = linearperturbations
        self.limber_dtype = {"double": None, "mixed": jnp.float32}[limber_precision]
        reference = None
        if nonlinear_model == "mead2020":
            if correction_reference is None:
                raise ValueError("nonlinear_model='mead2020' needs a correction_reference")
            if log10TAGN is not None:
                raise ValueError("log10TAGN is not a parameter of mead2020")
            reference = tuple(sorted((k, float(v)) for k, v in correction_reference.items()))
        table = _table(np.asarray(redshifts, float).tobytes(), background.mnu, nonlinear_model,
                       reference)
        self._table = table
        self.z, self.k = table.z, table.k
        self.log_t = MEAD2020_LOG_T_AGN if log10TAGN is None else log10TAGN
        h = background.h
        self._cosmo = dict(ombh2=background.Omega_b0 * h ** 2, omch2=background.Omega_cdm0 * h ** 2,
                           H0=background.H0, ns=background.ns, lnAs=jnp.log(1e10 * background.As),
                           background=background)
        lnp = table.lnp_table(self._cosmo, self.log_t)
        self.lnp, self._lnp_ia = lnp[:, :-1], lnp[:, -1]

    def _on_table(self, zs, ks):
        if np.shape(zs) != np.shape(self.z) or np.shape(ks) != np.shape(self.k):
            raise ValueError("TracedNonLinearPerturbations is evaluated on its table only")

    def log_matter_power_spectrum(self, zs, ks):
        self._on_table(zs, ks)
        return self.lnp

    def matter_power_spectrum(self, zs, ks):
        return jnp.exp(self.log_matter_power_spectrum(zs, ks))

    def limber_matter_power_spectrum(self, k_lz, z_l):
        """P(k_lz, z_l), shape ``(n_ell, len(z_l))``: see the module docstring."""
        lnp = self.lnp if self.limber_dtype is None else self.lnp.astype(self.limber_dtype)
        spline = _limber_maps(self.z.tobytes(), np.asarray(z_l, float).tobytes())
        if spline is not None:
            lnp = spline.astype(lnp.dtype) @ lnp
        lnk_u = np.log(self.k)
        dlnk = float((lnk_u[-1] - lnk_u[0]) / (len(lnk_u) - 1))
        lnk = jnp.log(k_lz).T
        x = (jnp.clip(lnk, lnk_u[0], lnk_u[-1]) - lnk_u[0]) / dlnk
        i = jnp.clip(jnp.floor(x), 1, len(lnk_u) - 3)
        x = i + (x - i).astype(lnp.dtype)  # the interval in float64, the weights in lnp's
        lnk = lnk.astype(lnp.dtype)
        value = lagrange4(lnp, x)
        hi = (lnp[:, -1] - lnp[:, -2])[:, None] / dlnk * jnp.maximum(lnk - float(lnk_u[-1]), 0.0)
        lo = (lnp[:, 1] - lnp[:, 0])[:, None] / dlnk * jnp.minimum(lnk - float(lnk_u[0]), 0.0)
        return jnp.exp(value + hi + lo).T

    def growth_factor(self, zs, ks=None):
        """D(z) (scale independent): sqrt[P(z, k_1)/P(0, k_1)], linear in P between the table's
        redshifts; above z = 5, D(5) D_cb(z)/D_cb(5)."""
        zs = np.asarray(zs, float)
        t = self._table
        i, w = linear_weights(t.z, zs)
        p = jnp.exp(self._lnp_ia - self._lnp_ia[0])
        d = jnp.sqrt((1.0 - w) * p[i] + w * p[i + 1])
        tail = np.flatnonzero(zs > EMULATOR_Z_MAX)
        if len(tail):
            d5 = jnp.sqrt(p[t.n_emu - 1])
            d = d.at[tail].set(d5 * self.background.growth_cb(zs[tail]))
        return d

    def sigma8_0(self):
        """sigma8 of the ``lcdm-1mass-s8-fs8`` network at z = 0."""
        log_t = self.log_t if self._table.feedback else LINEAR_LOG_T_AGN
        row = _Table.row(self._cosmo, log_t, self._table.mnu)
        return network(self._table.sigma, row[None, :])[0, 0]
