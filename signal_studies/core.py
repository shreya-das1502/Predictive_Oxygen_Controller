"""Continuous-time illustrative control models and explicit finite-record metrics."""
from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class P:
    kp: float = 2.0

    def transfer(self):
        return np.array([self.kp]), np.array([1.0])

    def response(self, t, error):
        return signal.lsim(self.transfer(), error, t, interp=False)[1]


@dataclass(frozen=True)
class PI(P):
    ki: float = 8.0

    def transfer(self):
        return np.array([self.kp, self.ki]), np.array([1.0, 0.0])


@dataclass(frozen=True)
class PID(PI):
    kd: float = 0.08
    tau_d: float = 0.01

    def transfer(self):
        # Kp + Ki/s + Kd*s/(1 + tau_d*s): realizable filtered derivative.
        return (np.array([self.kp*self.tau_d+self.kd,
                          self.kp+self.ki*self.tau_d, self.ki]),
                np.array([self.tau_d, 1.0, 0.0]))


def plant():
    """G(s)=1/((0.2s+1)(0.05s+1)), normalized conditioning amplifier."""
    return np.array([1.0]), np.array([0.01, 0.25, 1.0])


def loop(controller):
    cn, cd = controller.transfer()
    gn, gd = plant()
    return np.polymul(cn, gn), np.polymul(cd, gd)


def closed_loop(controller):
    """Reference T=GC/(1+GC); plant-input offset H=G/(1+GC)."""
    cn, cd = controller.transfer()
    gn, gd = plant()
    den = np.polyadd(np.polymul(cd, gd), np.polymul(cn, gn))
    return (np.polymul(cn, gn), den), (np.polymul(gn, cd), den)


def settling_time(t, error, tolerance, hold_s=1.0):
    """Earliest sampled time after last violation, with >=hold_s remaining.

    NaN means not established in this observation window, not infinity.
    """
    t, error = np.asarray(t), np.asarray(error)
    if t.ndim != 1 or len(t) != len(error) or len(t) < 2:
        raise ValueError('Need aligned time/error vectors')
    if tolerance <= 0 or hold_s < 0 or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid tolerance, hold duration or time order')
    if not np.isfinite(error).all():
        return np.nan
    bad = np.flatnonzero(np.abs(error) > tolerance)
    i = int(bad[-1]+1) if len(bad) else 0
    return float(t[i]-t[0]) if i < len(t) and t[-1]-t[i] >= hold_s else np.nan


def response_metrics(t, y, target=1.0):
    """Unit-reference metrics; tail is final one second, overshoot vs target."""
    t, y = np.asarray(t), np.asarray(y)
    return {'settling_2pct_s': settling_time(t, y-target, .02*abs(target)),
            'overshoot_pct': float(max(0, np.max(y)-target)/abs(target)*100),
            'tail_error': float(target-np.mean(y[t >= t[-1]-1]))}


def control_run(controller, dt=.002):
    t = np.arange(0, 20+dt/2, dt)
    reference = np.ones_like(t)
    disturbance = .3*(t >= 10)
    tracking, rejection = closed_loop(controller)
    nominal = signal.lsim(tracking, reference, t, interp=False)[1]
    delta = signal.lsim(rejection, disturbance, t, interp=False)[1]
    y = nominal+delta
    pre = t < 10
    metrics = response_metrics(t[pre], nominal[pre])
    equilibrium = controller.kp/(1+controller.kp) if type(controller) is P else 1.0
    metrics.update({'reference_equilibrium': equilibrium,
                    'analytic_reference_error': 1-equilibrium,
                    'settling_to_equilibrium_2pct_s': settling_time(t[pre], nominal[pre]-equilibrium, .02*equilibrium),
                    'overshoot_equilibrium_pct': float(max(0, np.max(nominal[pre])-equilibrium)/equilibrium*100)})
    post = t >= 10
    metrics.update({'disturbance_settling_2pct_s': settling_time(t[post], delta[post], .02*.3),
                    'disturbance_peak': float(np.max(np.abs(delta[post]))),
                    'disturbance_tail': float(np.mean(delta[t >= 19])),
                    'disturbance_iae': float(np.trapezoid(np.abs(delta[post]), t[post])),
                    'disturbance_tail_rejection_pct': float(100*(1-abs(np.mean(delta[t >= 19]))/.3)),
                    'post_step_tail_error': float(1-np.mean(y[t >= 19]))})
    return t, reference, disturbance, nominal, delta, y, metrics


def frequency_response(num, den, w):
    return np.polyval(num, 1j*np.asarray(w))/np.polyval(den, 1j*np.asarray(w))


def _positive_roots(coeff):
    coeff = np.trim_zeros(np.asarray(coeff), 'f')
    if len(coeff) < 2:
        return []
    roots = np.roots(coeff)
    return sorted({float(r.real) for r in roots if abs(r.imag) < 1e-6*max(1, abs(r.real)) and r.real > 1e-8})


def margins(num, den):
    """All finite positive crossovers from polynomial equations, not plot grid.

    Real rational strictly proper loops with no RHP zeros are supported here.
    Values at zero/infinity are excluded; limiting phase alone is no crossover.
    """
    num, den = np.asarray(num), np.asarray(den)
    # Polynomials in w: N(jw)*conj(D(jw)).
    nj = num*(1j**np.arange(len(num)-1, -1, -1))
    dj = den*(1j**np.arange(len(den)-1, -1, -1))
    power = np.polysub(np.polymul(nj, np.conj(nj)), np.polymul(dj, np.conj(dj))).real
    gain_cross = _positive_roots(power)
    cross = np.polymul(nj, np.conj(dj))
    phase_cross = [w for w in _positive_roots(cross.imag)
                   if frequency_response(num, den, w).real < 0]
    # Sum individual factor angles gives continuous phase starting at DC,
    # including integrator -90deg, without a finite-frequency unwrap grid.
    def phase(w):
        zeros, poles = np.roots(num), np.roots(den)
        return float(np.degrees(np.sum(np.angle(1j*w-zeros))-
                                np.sum(np.angle(1j*w-poles))))
    pms = [(180+phase(w), w) for w in gain_cross]
    gms = [(1/abs(frequency_response(num, den, w)), w) for w in phase_cross]
    pm, wgc = min(pms, key=lambda q: abs(q[0])) if pms else (np.nan, np.nan)
    gm, wpc = min(gms, key=lambda q: abs(np.log(q[0]))) if gms else (np.inf, np.nan)
    return {'gain_margin': float(gm), 'gain_margin_db': float(20*np.log10(gm)),
            'phase_margin_deg': pm, 'gain_crossover_rad_s': wgc,
            'phase_crossover_rad_s': wpc,
            'gain_crossover_count': len(gain_cross), 'phase_crossover_count': len(phase_cross)}, gain_cross, phase_cross
