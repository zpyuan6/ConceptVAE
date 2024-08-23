import matplotlib.pyplot as plt

def plot_generation_with_input(
    input_samples, 
    outputs,
    save_path: str = None
    ):

    fig, ax = plt.subplots(len(input_samples), 2)

    plt.axis('off')

    for i in range(len(input_samples)):
        ax[i][0].imshow(input_samples[i].permute(1, 2, 0).cpu().detach().numpy())
        ax[i][1].imshow(outputs[i].permute(1, 2, 0).cpu().detach().numpy())
        ax[i][0].axis("off")
        ax[i][1].axis("off")

    ax[0][0].set_title("Input")
    ax[0][1].set_title("Output")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_generation_with_title(
    sample_list: list, 
    title_list: list,
    save_path: str = None):

    fig, ax = plt.subplots(int(len(sample_list)//3)+1, 3)

    plt.axis('off')

    for i in range(len(sample_list)):
        ax[i//3][i%3].imshow(sample_list[i].permute(1, 2, 0).cpu().detach().numpy())
        ax[i//3][i%3].set_title(title_list[i])
        ax[i//3][i%3].axis("off")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


