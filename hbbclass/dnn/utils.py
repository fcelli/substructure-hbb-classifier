import matplotlib.pyplot as plt
import numpy


def plot_roc_curve(y_true, y_pred, xlims=(0., 1.)):
    n_sig = y_true[y_true==1].count()
    n_bkg = y_true[y_true==0].count()
    sig_eff = []
    bkg_rej = []
    for cut in numpy.linspace(0., 1., num=100):
        # Signal efficiency
        se = float(y_true[(y_true == 1) & (y_pred > cut)].count()) / n_sig
        # Background rejection
        br =  1 - (float(y_true[(y_true == 0) & (y_pred > cut)].count()) / n_bkg)
        sig_eff.append(se)
        bkg_rej.append(br)
    sig_eff = numpy.array(sig_eff, dtype=numpy.float32)
    bkg_rej = numpy.array(bkg_rej, dtype=numpy.float32)
    mask = [x > xlims[0] and x < xlims[1] for x in sig_eff]
    sig_eff = sig_eff[mask]
    bkg_rej = bkg_rej[mask]
    plt.xlabel('Signal Efficiency (TPR)')
    plt.ylabel('Background Rejection (1 - FPR)')
    plt.plot(sig_eff, bkg_rej)
