import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def save_percentage_plot(lstm_numbers, vae_numbers, combined_numbers, name):

    category_names = ['Correctly Classified', 'Misclassified', 'Other model correct, LSTM wrong', 'Both wrong']
    results = {
        'LSTM': lstm_numbers,
        'VAE': vae_numbers,
        'Combined': combined_numbers
    }


    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)

    color1 = np.asarray(colors.to_rgba('green'))
    color2 = np.asarray(colors.to_rgba('red'))
    category_colors = np.row_stack((color1, color2, color1, color2))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        if i > 1:
            colname = None
        ax.barh(labels, widths, left=starts, height=0.8,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            c = round(c,2)
            if c  > 0.025:
                ax.text(x, y, str(c), ha='center', va='center',
                        color=text_color)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                  loc='lower left', fontsize='small')

    plt.savefig(name +'.png')