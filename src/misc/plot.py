def train_instance_plot(temp_data_X, temp_data_Y, save_name):
    plt.figure(figsize=(12,8))
    plot_dx = 0
    
    for chdx in range(3):
        plt.plot(temp_data_X[:, chdx]/np.max(temp_data_X[:, chdx]) + plot_dx*2,color='k')
        plot_dx += 1
    
    plt.plot(temp_data_Y[:,0]-2,color='b')
    plt.plot(temp_data_Y[:,1]-3,color='r')
    plt.plot(temp_data_Y[:,2]-4,color='g')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_name ,dpi=300)
    plt.close()

    return