import matplotlib.pyplot as plt
import numpy


def plot_roc_curve(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    xlims: tuple[float, float] = (0., 1.),
    xlabel: str = 'Signal Efficiency (TPR)',
    ylabel: str = 'Background Rejection (1 - FPR)',
    ax: plt.axes = None,
    show_rnd_guess: bool = True,
    draw_grid: bool = True
) -> None:
    y_true = numpy.array(y_true)
    y_pred = numpy.array(y_pred)
    n_sig = len(y_true[y_true == 1])
    n_bkg = len(y_true[y_true == 0])
    sig_eff = []
    bkg_rej = []
    for cut in numpy.linspace(0., 1., num=100):
        # Compute true positive rate
        tp = len(y_true[(y_true == 1) & (y_pred > cut)])
        tpr = float(tp) / n_sig
        # Compute false positive rate
        fp = len(y_true[(y_true == 0) & (y_pred > cut)])
        fpr = float(fp) / n_bkg
        # Append signal efficiency
        sig_eff.append(tpr)
        # Append background rejection
        bkg_rej.append(1 - fpr)
    # Crop arrays based on x limits
    sig_eff = numpy.array(sig_eff, dtype=numpy.float32)
    bkg_rej = numpy.array(bkg_rej, dtype=numpy.float32)
    mask = [x >= xlims[0] and x <= xlims[1] for x in sig_eff]
    sig_eff = sig_eff[mask]
    bkg_rej = bkg_rej[mask]
    # Draw plot
    if ax is None:
        _, ax = plt.subplots()
    # Plot model ROC
    ax.plot(sig_eff, bkg_rej, label='Model')
    # Plot random guess ROC
    if show_rnd_guess:
        random_guess_x = numpy.linspace(0., 1., 100)
        random_guess_y = numpy.linspace(1., 0., 100)
        mask = [x >= xlims[0] and x <= xlims[1] for x in random_guess_x]
        random_guess_x = random_guess_x[mask]
        random_guess_y = random_guess_y[mask]
        ax.plot(random_guess_x, random_guess_y, label='Random guess', linestyle='dashed')
    # Set axes titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Set limits
    ax.set_xlim(xlims)
    y_min = min(min(random_guess_y), min(bkg_rej))
    y_max = max(max(random_guess_y), max(bkg_rej))
    ax.set_ylim((y_min, y_max))
    # Draw legend
    ax.legend()
    # Draw grid
    if draw_grid:
        ax.grid()
