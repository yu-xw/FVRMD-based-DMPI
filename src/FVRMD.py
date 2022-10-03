#%%
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import time

from utils.utils import variational_operator, err_in_percent, construct_component

# joint, adaptive seg_length
def FVRMD(y, fs, damp_r, freq, n_seg, 
          alpha=1e2, beta=0.5, gamma=10, 
          delta_damp_r=0.01, tau=1e-8, r=1, 
          delta_diff=1e-8,delta_reg=1e-8, max_iters=100, 
          seg=(1/4, 3/4), quite=False):
    """ FVRMD algorithm, for full parameters (frequency, damping ratio, initial
    amplitude, initial phase) identification from multi-DOF free vibration 
    responses.

    Parameters:
    ----------
    `y`: 1-d array
        Free vibration response, damped signal.
    `fs`: float or int
        Sampling rate.
    `damp_r`: 
        Initial damping ratio.
    `freq`:
        Initial frequency.
    `n_seg`:
        Signal length for component.
    `alpha`:
        Initial value of penalty factor. Defaults to 1e2.
    `beta`:
        Factor to control the step of updating frequency and damping ratio. 
        Defaults to 0.5.
    `gamma`:
        Parameter to control the increase of `alpha`. Defaults to 10.
    `delta_damp_r`:
        Parameter to control the max step of updating damping ratio. 
        Defaults to 0.01.
    `tau`:
        Factor to Lagrangian multiplier. Defaults to 1e-8.
    `r`:
        Order of variation. Defaults to 1.
    `delta_diff`:
        Stop creteria of difference convergence. Defaults to 1e-8.
    `delta_reg`:
        Stop creteria of regularization/variation convergence. Defaults to 1e-8.
    `max_iters`:
        Maximum numbers of iterations. Defaults to 100.
    `seg`: tuple
        To intercept a segment from a sequence. Defaults to (1/4, 3/4).
    `quite`: bool
        If print info. during iterations. Defaults to False.

    Returns:
    -------
    `Res`: dict
        {
            'Y':             # decomposed signal comp.
            'Y_rec':         # reconstruction error
            'amp':           # initial amplitude
            'damp_r':        # damping ratio, in %
            'freq':          # frequency, in Hz
            'phase':         # initial phase, in rad/s
        }
    `Log`: dict, record the intermediate results during iterations
        {
            'loss_rec':     # reconstruction err
            'loss_reg':     # varying criteria of regularization convergence 
            'diff':         # varying criteria of difference convergence 
            'alpha':        # varying frequency 
            'freq':         # varying frequency 
            'damp_r':       # varying damping ratio 
        }
    """
    damp_r = damp_r.copy()
    freq = 2*np.pi*freq.copy()
    alpha_init = alpha

    n = np.max(n_seg)
    y = y[:n]
        
    K = len(freq)
    t = np.linspace(0, n/fs, n, endpoint=False)
    
    # pre_calculate
    T = [variational_operator(ns, r) for ns in n_seg]
    D = [sp.csc_array(np.block([[T, np.zeros_like(T)],
                                [np.zeros_like(T), T]])) for T in T]
    DTD = [D.T @ D for D in D]
    
    ts = [np.linspace(0, ns/fs, ns, endpoint=False) for ns in n_seg]
    B = []
    for k in range(K):
        ns = n_seg[k]
        tt = ts[k][int(ns*seg[0]):int(ns*seg[1])]
        temp = np.array([tt, np.ones_like(tt)]).T
        B.append(la.inv(temp.T @ temp) @ temp.T)
    
    eps = np.finfo(float).eps

    Log = {
        'loss_rec': np.zeros((max_iters, 1)), # reconstruction err.
        'loss_reg': np.zeros((max_iters, K)), # reg. err. of each comp.
        'diff': np.zeros((max_iters, K)),
        'alpha': np.zeros((max_iters, K)),
        'freq': np.zeros((max_iters, K)),
        'damp_r': np.zeros((max_iters, K))
    }

    # init.
    Y, Y_old = np.zeros((n, K)), np.zeros((n, K))
    X = np.zeros((2*n, K))
    lambd = np.zeros(n)
    amp = np.zeros(K)
    phase = np.zeros(K)
    diff = np.zeros(K)
    loss_reg = np.zeros(K)
    A = [(construct_A(damp_r[k], freq[k], ts[k])) for k in range(K)]  # sparse

    # main loop
    alpha = alpha_init * np.ones(K)
    inx = np.arange(K, dtype=int)
    rk = np.arange(K, dtype=int)
    jj = 0
    for i in range(max_iters):
        jj += 1
        for k in rk:
            ns = n_seg[k]
            #TODO: fast matrix inversion
            temp = alpha[k] * DTD[k] + A[k].T @ A[k]
            temp_inv = sla.spsolve(temp, np.eye(2*ns))
            b = y[:ns] - np.sum(Y[:ns, inx != k], axis=1)
            X[:2*ns, k] = temp_inv @ (A[k].T @ (b + lambd[:ns]))  # dense

            # update damp_r_k and f_k
            aa = int(ns*seg[0])
            bb = int(ns*seg[1])
            temp = np.unwrap(np.angle(X[:ns, k] - X[ns:2*ns, k]*1j))
            ff = B[k][0] @ temp[aa:bb]
            freq[k] += beta * ff
        
            ss = X[:ns, k]**2 + X[ns:2*ns, k]**2
            temp = - 0.5 * np.log(ss) / freq[k]
            dd = beta * B[k][0] @ temp[aa:bb]
            dd = np.sign(dd) * np.min([np.abs(dd), delta_damp_r])
            damp_r[k] += dd
            damp_r[k] = np.max([damp_r[k], 0])
            
            # update A_k
            A[k] = construct_A(damp_r[k], freq[k], ts[k])
            
            # calculate amplitude and phase
            amp[k] = np.mean(np.sqrt(ss)[aa:bb])
            phase[k] = np.mean(np.unwrap(np.angle(X[:ns, k]-X[ns:2*ns, k]*1j))[aa:bb])
            # update y_k
            Y[:, k] = construct_component(
                amp[k], damp_r[k], freq[k], phase[k], t)
            
            # convergence criterion
            diff[k] = (la.norm(Y[:ns, k] - Y_old[:ns, k], axis=0)**2) / \
                (eps + la.norm(Y_old[:ns, k], axis=0)**2)
            loss_reg[k] = la.norm(D[k] @ X[:2*ns, k], axis=0) ** 2 / ns

        loss_rec = la.norm(np.sum(Y, axis=1) - y) ** 2
        # update lambd
        lambd += tau * (y - np.sum(Y, axis=1))
            
        # adaptive growth of `alpha`
        # alpha0 *= alpha_rate
        # gamma = gamma * la.norm(y)**2 / loss_rec
        rate = 10**(2*np.arctan(jj/gamma))
        # rate = (np.exp(np.log(10**3)/(3+np.pi/2))) ** (np.arctan((jj-20)/2)+3)
        alpha[rk] = rate * alpha_init
        # rr = [(la.norm(Y[:n_seg[k], k])**2 / la.norm(y[:n_seg[k]] - Y[:n_seg[k], k])**2) for k in rk]
        # rr = 1/amp[rk] * freq[rk] * damp_r[rk]
        # rr /= np.max(rr)
        # rr = 1
        # alpha[rk] *= rr

        # intermediate output
        if not quite:
            print(f'iters-{i}-----------------------')
            print(f'd_ramping={np.floor(damp_r*100000)/100000}')
            [print(f'diff[{j}]={diff[j]:.4e}', end=', ') for j in rk]
            print('')
            [print(f'loss_reg[{j}]={loss_reg[j]:.4e}', end=',') for j in rk]
            print(f'loss_rec={loss_rec:.4e}\n')

        # Logging
        Log['loss_rec'][i] = loss_rec  # residual signal
        Log['loss_reg'][i, rk] = loss_reg[rk]
        Log['diff'][i, rk] = diff[rk]
        Log['alpha'][i, rk] = alpha[rk]
        Log['freq'][i, rk] = freq[rk]
        Log['damp_r'][i, rk] = damp_r[rk]

        if i >= max_iters:
            break
        
        for k in rk:
            if loss_reg[k] < delta_reg or diff[k] < delta_diff:
                rk = np.delete(rk, np.where(rk == k)[0][0])  # exclude
                for item in list(Log.keys())[1:]:
                    Log[item][i+1:, k] = None
                # alpha[rk] = alpha_init
                # jj = 0
                # gamma = 100
        
        if len(rk) == 0:
            break

        Y_old = Y.copy()
        i += 1

    # truncate
    for item in Log: Log[item] = Log[item][:i+1]
               
    Y_rec = np.zeros((n, K))
    for k in range(K):
        Y_rec[:, k] = construct_component(amp[k], damp_r[k], freq[k], phase[k], t)
    
    Res = {
        'Y': Y,  # decomposed signal comp.
        'Y_rec': Y_rec, # reconstruction
        'amp': amp,
        'damp_r': damp_r,
        'freq': freq/2/np.pi,
        'phase': phase
    }
    return Res, Log

def construct_A(damp_r, freq, t):
    # construct A
    factor = freq * damp_r
    C = np.diag(np.exp(-factor*t)*np.cos(freq*t))
    S = np.diag(np.exp(-factor*t)*np.sin(freq*t))
    A = np.block([C, S])
    return sp.csc_array(A)

#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    from gen_signal import *
    from ERA import ERA
    
    fs = 100
    K = 3
    sig = get_damped_signal(K, T=20, fs=fs, noise_std=1, seed=324)
    
    
    Res_ERA = ERA(sig['y'][:500].reshape(-1, 1), nin=1, order=K, fs=fs)
    Res = Res_ERA
    print('Results by ERA:')
    print(pd.DataFrame({
        "Damping ratio": Res['damp_r'],
        "Error_d": err_in_percent(sig['damp_r'], Res['damp_r']),
        "Freqency": Res['freq'],
        "Error_f (%)": err_in_percent(sig['freq'], Res['freq'])
    }))

    freq = Res_ERA['freq']
    damp_r = np.zeros_like(freq)
    
    #%%
    t1 = time.time()
    a0 = 1/freq * fs * 10
    a =  1/freq * fs * (-np.log(0.25) / (Res_ERA['damp_r']*2*np.pi))
    a1 = 1/freq * fs * 15
    print(pd.DataFrame({
        'a0': a0,
        "a": a,
        "a1": a1
    }))
    a[a<a0] = a0[a<a0]
    a[a>a1] = a1[a>a1]
    n_seg = np.array(a, dtype=int)
    print(f'segments are {n_seg}')
    Res_FVRMD, Log = FVRMD(sig['y'], fs=fs, damp_r=damp_r, freq=freq,
                            n_seg=n_seg, alpha=1e2, beta=0.5,
                            delta_damp_r=0.01,
                            seg=(0, 1), gamma=20, quite=True)
    t2 = time.time() - t1
    
    Res = Res_FVRMD
    print('Results by FVRMD:')
    print(pd.DataFrame({
        "amplitude": Res['amp'],
        "Err_a": err_in_percent(sig['amp'], Res['amp']),
        "damping_ratio": Res['damp_r'],
        "Err_d": err_in_percent(sig['damp_r'], Res['damp_r']),
        "freqency": Res['freq'],
        "Err_f": err_in_percent(sig['freq'], Res['freq']),
        "phase": Res['phase'],
        "Err_p": err_in_percent(sig['phase'], Res['phase']),
    }))
    print(f'Total cost {t2:.4f} s')

    #%%
    K = len(Res['freq'])
    t = sig['time']
    fig, axs = plt.subplots(K+1, 1, tight_layout=True)
    for i in range(K):
        axs[i].plot(Res['Y'][:, i], '--', lw=1, label='extracted')
        # axs[i].plot(Res['Y_rec'][:, i], '-.', lw=1, label='reconstructed')
        temp = sig['Y'][:, i]
        axs[i].plot(temp, '-.', lw=1, label='truth')
        axs[i].legend()
        axs[i].set_title(f'component {i}')
    axs[K].plot(np.sum(Res['Y'], axis=1), label='resc')
    axs[K].plot(sig['y'], label='noisy')
    axs[K].plot(sig['y'] - sig['noise'], '--', label='original')
    axs[K].legend()
        
    # damp
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, tight_layout=True)
    # lns = ax1.plot(np.log(Log['damp']), marker='.')
    lns = ax1.plot(Log['damp_r'], marker='.')
    [lns[i].set_color(c) for i, c in enumerate(('r', 'g', 'b'))]
    lns = ax1.plot(np.ones_like(Log['damp_r']) * sig['damp_r'], '--')
    [lns[i].set_color(c) for i, c in enumerate(('r', 'g', 'b'))]
    # ax1.legend([1, 2, 3, -1, -2, -3])
    ax1.grid()
    ax1.set_title('varying damp ratio')
    lns = ax2.plot((Log['alpha']), marker='.')
    [lns[i].set_color(c) for i, c in enumerate(('r', 'g', 'b'))]
    ax2.grid()
    ax2.set_title('varying alpha')
    lns = ax3.plot(-np.log10(Log['loss_reg']), '--', marker='.')
    [lns[i].set_color(c) for i, c in enumerate(('r', 'g', 'b'))]
    # ax3.legend([1, 2, 3, -1, -2, -3])
    ax3.grid()
    ax3.set_title('varying loss_reg')
    plt.show()

