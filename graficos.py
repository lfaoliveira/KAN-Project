import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_dataframes_comparison(df_mlp: pd.DataFrame, df_kan: pd.DataFrame,
                               title_mlp, title_kan, column_names: list, path_save):
    """
    Plot a column from two different dataframes as subplots with threshold markers.

    Parameters:
    df1, df2: pandas DataFrames
    column_name: string, the name of the column to plot
    title1, title2: strings, titles for each subplot
    """
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    threshold = 100  # You can adjust this threshold value

    # Rename 'MAE' to 'val_loss' in both dataframes
    df_mlp = df_mlp.rename(columns={'MAE': 'val_loss'})
    df_kan = df_kan.rename(columns={'MAE': 'val_loss'})

    # Add red dots for values above threshold
    for col in column_names:
        above_threshold = df_mlp[df_mlp[col] > threshold]
        ax1.scatter(above_threshold.index,
                    above_threshold[col], color='red', zorder=5)
        df_mlp.drop(above_threshold.index)
    df_mlp[column_names].plot(ax=ax1)

    ax1.set_title(title_mlp)
    ax1.set_xlabel('Epoch')
    ax1.set_ylim(top=threshold, bottom=0)
    ax1.set_ylabel('Loss')

    # Plot second DataFrame
    # Add red dots for values above threshold
    for col in column_names:
        above_threshold = df_kan[df_kan[col] > threshold]
        ax2.scatter(above_threshold.index,
                    above_threshold[col], color='red', zorder=5)
        df_kan.drop(above_threshold.index)

    df_kan[column_names].plot(ax=ax2)
    ax2.set_title(title_kan)
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(top=threshold, bottom=0)
    ax2.set_ylabel('Loss')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.legend()
    plt.savefig(path_save)
    # plt.show()


lista_csv_mlp = filter(lambda x: f"csv" in x and "MLP" in x,
                       os.listdir(os.getcwd()))

lista_csv_kan = filter(lambda x: f"csv" in x and "KAN" in x,
                       os.listdir(os.getcwd()))

lista_mlp = list(lista_csv_mlp)
lista_kan = list(lista_csv_kan)
a = "RESUMO_KAN.csv"
b = "RESUMO_MLP.csv"

if a in lista_kan:
    lista_kan.remove(a)
if b in lista_mlp:
    lista_mlp.remove(b)

df_final = None
fold = 1
df_resumo = []
for path_mlp, path_kan in zip(lista_mlp, lista_kan):
    df_mlp = pd.read_csv(path_mlp, index_col=0, decimal=",")
    df_kan = pd.read_csv(path_kan, index_col=0, decimal=",")

    title_mlp = f"FOLD_{fold} MLP"
    title_kan = f"FOLD_{fold} KAN"
    plot_dataframes_comparison(
        df_mlp, df_kan, title_mlp, title_kan, column_names=["train_loss", "val_loss"], path_save=f"LOSS_FOLD_{fold}")

    fold += 1
