import itertools
import numpy as np
import numpy.linalg as la


def ERA(Y, nin, order, fs):
    """ 
    `ERA` algorithm, translated from the matlab code. Some changes are made.

    Parameters
    ----------
    `Y`: array_like
        Free vibration output data in a form of Y=[Y1 Y2 ... Y_Ntime] Yi is 
        Markov Parameter of size (nout, nin) and the total size is 
        (nout, nin*Ntime), where outputs is the number of output channels, 
        inputs is the number of inputs which equals to 1 unless free vibration 
        data comes from Multi-reference channels NExT. Ntime is the length of 
        the data samples.
    `nin` : 
        The number of inputs which equals to 1 unless free vibration data
        comes from Multi-reference channels NExT.
    `order`: int
        Number of modes.
    `fs`: float or int
        Sampling frequency.
        
    Returns
    -------
    dict:
        {
            "freq":         in Hz
            "damp_r":       in %
            "mode":         channels x orders
        }
    """

    r = 2 * order
    # check if Y should be transposed
    [nout, npts] = Y.shape
    if nout > npts:
        Y = Y.T
        [nout, npts] = Y.shape

    # reshape Y from (nout, nin*Ntime) to (Ntime, nout, nin) to
    # construct the generalized Hankel matrices easier
    Ntime = int(npts/nin)
    Y = np.stack(np.split(Y, Ntime, axis=1))

    # Construct Hankel matrices H0 and H1
    ncols = int(2/3 * Ntime)
    nrows = Ntime - ncols + 1

    H = np.zeros((nrows*nout, ncols*nin))
    for i, j in itertools.product(range(nrows), range(ncols)):
        H[i*nout:(i+1)*nout, j*nin:(j+1)*nin] = Y[i+j]
    H0 = H[:, :(ncols-1)*nin]
    H1 = H[:, nin:]

    # decompose the data matrix
    U, s, vH = la.svd(H0, full_matrices=False)

    # truncate the matrices using the cutoff
    D = np.diag(np.sqrt(s[:r]))  # square root of the singular values
    D_inv = la.inv(D)  # (sigma)^(-1/2)
    Ur = U[:, :r]
    Vr = vH.conjugate().T[:, :r]

    A = D_inv @ Ur.conjugate().T @ H1 @ Vr @ D_inv
    B = (D @ Vr.conjugate().T)[:, :nin]
    C = (Ur @ D)[:nout, :]
    del H, H0, H1

    # extract frequency, damping ratio and mode shape
    values, vectors = la.eig(A)
    s = np.log(values)  # laplace roots
    f = s.imag
    freq = np.sort(f)[order:] * fs  # damped frequency in rad/s
    inx = np.argsort(f)[order:]
    damp_r = -s[inx].real / np.abs(s[inx])  # damping ratio
    mode = C @ vectors[:, inx]  # mode shape

    Phi = np.zeros(mode.shape)
    v = np.max(np.abs(mode), axis=0)
    v_inx = np.argmax(np.abs(mode), axis=0)
    for i in range(len(v)):
        b = -np.angle(mode[v_inx[i], i])
        Phi[:, i] = np.real(mode[:, i] * np.exp(1j*b))
        Phi[:, i] = Phi[:, i] / la.norm(Phi[:, i])

    return {
        "freq": freq/2/np.pi,
        "damp_r": damp_r,
        "mode": Phi # channels * orders
    }

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    from gen_signal import get_damped_signal, plot_damped_signal

    fs = 100
    sig = get_damped_signal(3, T=3, fs=fs, noise_std=0, seed=324)
    
    Res = ERA(sig['y'].reshape(-1, 1), nin=1, order=3, fs=fs)
    def err(x_t, x): return (x - x_t) / x_t * 100

    print(pd.DataFrame({
        "Damping ratio": Res['damp_r'],
        "Error_d": err(sig['damp_r'], Res['damp_r']),
        "Freqency": Res['freq'],
        "Error_f (%)": err(sig['freq'], Res['freq'])
    }))
    

