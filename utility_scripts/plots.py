def tsne_plot(outputs):
    """
    Creates a labelled TSNE plot given the ge2e model outputs
    :param outputs: Tensor of model outputs
    """
    from sklearn.manifold import TSNE
    import numpy as np
    import matplotlib.pyplot as plt

    emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    tsne = TSNE()
    outputs_np = outputs.detach().numpy()
    i, j, k = outputs_np.shape
    tsne_input = np.empty([i*j, k])
    c = 0
    for ii in range(i):
        for jj in range(j):
            tsne_input[c] = outputs_np[ii, jj, :]
            c += 1
    tsne_out = tsne.fit_transform(tsne_input)

    fig, ax = plt.subplots(figsize=[10, 10])
    for p in range(i):
        plt.scatter(tsne_out[p*j:p*j+j, 0], tsne_out[p*j:p*j+j, 1], label=emotions[p])
    plt.legend(prop={'size': 15})
    plt.title('TSNE plot of model outputs')
