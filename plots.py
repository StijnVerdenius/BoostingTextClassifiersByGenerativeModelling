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
            c = round(c, 2)
            if c > 0.025:
                ax.text(x, y, str(int(c*100)), ha='center', va='center',
                        color=text_color, size='xx-large')
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                  loc='lower left', fontsize='small')

    plt.savefig(name +'.png')

def save_lineplot_guan(classifier_accs, vae_acss, combined_accs, name):
    fig = plt.figure()
    ax = plt.axes()

    ys = [[],[],[]]
    for i, accs_dict in enumerate([classifier_accs, vae_acss, combined_accs]):
        x = []
        for k, v in accs_dict.items():
            x.append(k)
            ys[i].append(v)

    ax.plot(x, ys[0],label='LSTM')
    ax.plot(x, ys[1],label='VAE')
    ax.plot(x, ys[2],label='Combined')
    ax.legend()
    plt.savefig(name +'.png')

def save_lineplot_per_genre(data):
    genres = ['Pop', 'HipHop', 'Rock', 'Metal', 'County']
    model = ['LSTM', 'VAE','Combined']
    plot_data = {}

    for genre_i, genre in enumerate(genres):
        plot_data[genre] = {}
        for i, accs_dict in enumerate(data[genre_i][0]):
            temp_x = []
            temp_y = []
            for k, v in accs_dict.items():
                temp_x.append(k)
                temp_y.append(v)
            plot_data[genre][model[i]] = temp_y
        plot_data[genre]['x'] = temp_x

    for i, name in enumerate(model):
        fig = plt.figure()
        ax = plt.axes()
        for genre_i, genre in enumerate(genres):
            ax.plot(plot_data[genre]['x'], plot_data[genre][name],label=genre)
        ax.legend()
        plt.savefig(name + '.png')
