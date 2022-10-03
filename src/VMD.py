import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import hilbert
from utils.utils import get_params_from_monoharmonic, err_in_percent

def VMD(sig, fs, alpha, tau, K, DC, init, freq0, tol, fix_freq=False):
    """ VMD, translated from the matlab code. Some changes are made.
         
    Parameters:
    ----------
    `sig`:
        the time domain signal (1D) to be decomposed
    `fs`:
        sampling rate
    `alpha`:
        the balancing parameter of the data-fidelity constraint
    `tau`:
        time-step of the dual ascent (pick 0 for noise-slack)
    `K`:
        the number of modes to be recovered
    `DC`:
        true if the first mode is put and kept at DC (0-freq)
    `init`:
        0 = all omegas start at 0, 
        1 = all omegas start uniformly distributed,
        2 = all omegas initialized randomly
        else = specified initial values `freq0`
    `freq0`:
        initial frequency by specified
    `tol`:
        tolerance of convergence criterion; typically around 1e-6
    `fix_freq`:
        if optimize frequency, `False` the frequency is fixed as the initial 
        values. If `init=0, 1, or 2`, fix_freq should False.   
             
    Returns:
    -------
    dict :
        {
            "Y":          # decomposed comp. 
            "Y_hat":      # spectrum of decomposed comp.
            "freq":       # optimized frequency
            "freq_path":  # record of frequency during iteration
            "amp":        # amplitude estimated from decomposed comp.
            "freq2":      # frequency estimated from decomposed comp.
            "damp_r":     # damping ratio estimated from decomposed comp.
            "phase":      # phase estimated from decomposed comp.
        }
        
    """
    # extend the signal by mirroring
    N = len(sig)
    sig_m = np.zeros(4*N//2 - N%2)  # odd for 2*N-1, even for 2*N
    sig_m[:N//2] = np.flip(sig[:N//2])
    sig_m[N//2:N//2+N] = sig
    sig_m[N//2+N:] = np.flip(sig[N//2 + N%2:])

    # spectral domain discretization, normalized frequency bins
    N_m = len(sig_m)
    freq = rfftfreq(N_m)
    N_freq = len(freq)
    # Maximum number of iterations (if not converged yet, then it won't anyway)
    max_iters = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)
    # Construct f_hat
    f_hat = rfft(sig_m)

    # Initialization of omega_k
    freq_plus = np.zeros((max_iters, K))
    if init == 0:
        freq_plus[0] = np.zeros((K))
    elif init == 1:
        freq_plus[0] = np.arange(0, K, 1) / K * 0.5
    elif init == 2:
        freq_plus[0] = np.sort(np.exp(np.log(1/N)) +
                                (np.log(0.5)-np.log(1/N)) * np.random.rand(K))
    else:
        freq_plus[0] = freq0

    # if DC mode imposed, set its omega to 0
    if DC:
        freq_plus[:, 0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros(N_freq)
    # other inits
    eps = np.finfo(float).eps
    uDiff = tol + eps  # update step
    n = 0  # loop counter
    u_hat = np.zeros((N_freq, K), dtype=complex)
    u_hat_old = np.zeros((N_freq, K), dtype=complex)

    # Main loop for iterative updates
    inx = np.arange(K)
    while (uDiff > tol and n + 1 < max_iters):
        for k in inx:
            u_hat[:, k] = (f_hat - np.sum(u_hat[:, inx != k], axis=1) -
                lambda_hat/2) / (1 + Alpha[k] * (freq - freq_plus[n, k])**2)
            if fix_freq:
                freq_plus[n+1, k] = freq0[k]
            elif k == 0 and ~DC:
                spec_2 = np.abs(u_hat[:, k])**2
                freq_plus[n+1, k] = (freq @ spec_2) / np.sum(spec_2)
            elif k > 0:
                spec_2 = np.abs(u_hat[:, k])**2
                freq_plus[n+1, k] = (freq @ spec_2) / np.sum(spec_2)

        # Dual ascent
        lambda_hat = lambda_hat + tau*(np.sum(u_hat, axis=1) - f_hat)
        # loop counter
        n += 1
        uDiff = eps + 1/N_freq * np.linalg.norm(u_hat - u_hat_old)**2
        uDiff = np.abs(uDiff)

    # ------ Postprocessing and cleanup
    # discard empty space if converged early
    M = min(max_iters, n)
    freq = freq_plus[:M, :]

    # signal reconstruction
    u = irfft(u_hat, axis=0).real
    # remove mirror part
    u = u[N//2: N//2+N]

    # recomputing spectrum
    u_hat = rfft(u, axis=0) * 2 / N

    # sort
    inx = np.argsort(freq[-1])
    f = np.sort(freq[-1]) * fs
    u = u[:, inx]
    u_hat = u_hat[:, inx]

    # obtain 'damping ratio' and 'initial amplitude' and 'initial phase'
    amp = np.zeros(K)
    damp_r = np.zeros(K)
    freq2 = np.zeros(K)
    phase = np.zeros(K)
    
    t = np.linspace(0, N/fs, N, endpoint=False)
    for k in np.arange(K):
        amp[k], damp_r[k], freq2[k], phase[k] = get_params_from_monoharmonic(u[:, k], t)

    return {
        "Y": u, 
        "Y_hat": u_hat,
        "freq": f, # optimized frequency
        "freq_path": freq,
        "amp": amp,
        "freq2": freq2/2/np.pi,
        "damp_r": damp_r,
        "phase": phase
    }
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from gen_signal import get_damped_signal, plot_damped_signal
    
    fs = 100
    sig = get_damped_signal(3, T=5, fs=fs, noise_std=0, seed=321)
    plot_damped_signal((15/2.54, 4/2.54), sig)
    print(f'True frequency: {sig["freq"]}')
    
    res = VMD(sig['y'], fs=fs, alpha=2000, tau=1e-6, K=3, DC=0, init=1, 
              freq0=None, tol=1e-8, fix_freq=False)
    
    print(pd.DataFrame({
        "amplitude": res['amp'],
        "Err_a": err_in_percent(sig['amp'], res['amp']),
        "damping_ratio": res['damp_r'],
        "Err_d": err_in_percent(sig['damp_r'], res['damp_r']),
        "freqency": res['freq'],
        "Err_f": err_in_percent(sig['freq'], res['freq']),
        "freqency2": res['freq2'],
        "Err_f2": err_in_percent(sig['freq'], res['freq2']),
        "phase": res['phase'],
        "Err_p": err_in_percent(sig['phase'], res['phase'])
    }))
    
    u, u_hat, freq, *_ = res.values()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15/2.54, 5/2.54), tight_layout=True)
    n = len(sig['y'])
    ax1.plot(rfftfreq(n, 1/fs), np.abs(u_hat))
    ax1.plot(rfftfreq(n, 1/fs), np.abs(rfft(sig['y']))*2/n, '--')
    ax1.plot(np.array([freq, freq]), np.array([np.zeros(len(freq)),
                np.max(np.abs(u_hat))*np.ones(len(freq))]), '--', c='#2e2e2e')
    ax2.plot(sig['time'], u)
    ax2.plot(sig['time'], sig['Y'], '--')
    plt.show()
