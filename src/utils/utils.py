import numpy as np
from numpy.random import default_rng
import scipy.linalg as la
import scipy.sparse as sp
from scipy.signal import hilbert

def variational_operator(n, r):
    """Construct a (first or second order) variational operator 
    with size of (n-r)*n

    Parameters:
    ----------
    `n`: Number of columns.
    `r`: Order, should be 1 or 2.

    Returns:
    `F`: Total variation operator, size is (n-r)*n.
    """
    ex = np.ones(n)
    if r == 1:
        data = np.array([-ex, ex])
        offsets = np.array([0, 1])
        return sp.dia_array((data, offsets), shape=(n-1, n)).toarray()
    elif r == 2:
        data = np.array([ex, -2*ex, ex])
        offsets = np.array([-1, 0, 1])
        arr = sp.dia_array((data, offsets), shape=(n, n)).toarray()
        return arr[1:-1]

def lse_line(y, x):
    """Least square estimate of the parameters of a line
    y = ax + b = [x, 1] * [a, b].T
    """
    X = np.array([x, np.ones_like(x)]).T
    return np.linalg.inv(X.T @ X) @ X.T @ y

def err_in_percent(x_t, x): 
    """Compute the relative error in percent.
    
    Parameters:
    ----------
    `x_t`: The true value.
    `x`: The estimated value.        
    """
    return (x - x_t) / x_t * 100

# calculate snr
def snr(sig, noise): 
    """Compute signal-to-noise-ratio (snr).

    Parameters:
    ----------
    `sig`: Signal
    `noise`: Noise

    Returns:
    -------
    snr: SNR in dB
    """
    return 10 * np.log10(la.norm(sig)**2 / la.norm(noise)**2)

# white noise
def get_noise(size, noise_std, seed):
    rng = default_rng(seed)
    return rng.normal(0, noise_std, size)

def construct_component(amp, damp_r, freq, phase, t):
    return amp * np.exp(-freq * damp_r * t) * np.cos(freq * t + phase)

def get_params_from_monoharmonic(u, t, seg=(1/3, 2/3)):
    N = len(u)
    a = int(seg[0] * N)
    b = int(seg[1] * N)
    u_ht = hilbert(u)
    u_ang = np.unwrap(np.angle(u_ht)) # unwrapped angle in rad
                                      # ang = freq*t + phi
    freq, phase = lse_line(u_ang[a:b], t[a:b])
    u_env = np.abs(u_ht)  # envelope: ln(env) = -sigma_r*t + ln(a)
    p = lse_line(np.log(u_env[a:b]), t[a:b])
    damp_r = -p[0] / freq
    amp = np.exp(p[1])
    
    return amp, damp_r, freq, phase  # freq in rad/s

def params_assemble(amp, damp_r, freq, phase, ref=0, method='max'):
    """Amplitude based parameters integration: determine frequency and damping 
    ratio; determine modal shape.

    Parameters
    ----------
    `amp`: array_like, orders * channels
        Amplitude matrix.
    `damp_r`: array_like, orders * channels
        Damping ratio matrix.
    `freq`: array_like, orders * channels
        Frequency matrix.
    `phase`: array_like, orders * channels
        Phase matrix.
    `ref`: int
        Reference channel, default to 0.
    `method`: str, default to 'max'.
        Method to determine freqency and damping ratio:
        - 'max' for maximum amplitude based method,
        - 'ave1' for weighted average method 1,
        - 'ave2' for weighted average method 2.
        
    Returns:
    -------
    dict:
        {
            "freq":         # frequency
            "damp_r":       # damping ratio,
            "mode":         # modal shape
        }
    
    """
    THRES = np.pi/2
    
    diff_phase = phase - phase[ref]
    sign = np.zeros_like(phase)
    sign[np.abs(diff_phase) <= THRES] = 1
    sign[np.abs(diff_phase) > THRES] = -1
    
    mode = amp / amp.max(axis=0) * sign
    K = amp.shape[1]
    
    if method == 'max':
        # frequency and damping ratio selected by max amplitude
        freq = [freq[amp[:, i] == amp[:, i].max(), i][0] for i in range(K)]
        damp_r = [damp_r[amp[:, i] == amp[:, i].max(), i][0] for i in range(K)]
    elif method == 'ave1':
        # frequency and damping ratio selected by average
        freq = [freq[:, i].T @ amp[:, i] / amp[:, i].sum() for i in range(K)]
        damp_r = [damp_r[:, i].T @ amp[:, i] / amp[:, i].sum() for i in range(K)]
    elif method == 'ave2':
        A = (amp.T/amp.sum(axis=1).T).T # order-wise, ratio between orders
        B = amp / amp.sum(axis=0) # channel-wise, ratio between channels
        W = A * B
        freq = [freq[:, i].T @ W[:, i] / W[:, i].sum() for i in range(K)]
        damp_r = [damp_r[:, i].T @ W[:, i] / W[:, i].sum() for i in range(K)]
    
    
    return {
        "freq": freq,
        "damp_r": damp_r,
        "mode": mode
    }
    
    
def get_MAC(phi_t, phi_e):
    """Compute the MAC value of two modal shape

    Parameters:
    ----------
    phi_t: modal shape a
    phi_e: modal shape b
    
    """
    return np.inner(phi_t, phi_e) ** 2 / (np.inner(phi_t, phi_t) * 
                                          np.inner(phi_e, phi_e))