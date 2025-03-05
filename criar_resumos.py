import os
import pandas as pd


MODELO = "MLP"
FINAL = False
path_df_final = f"FINAL_{MODELO}.csv"
path_df_resumo = f"RESUMO_{MODELO}.csv"

lista_csvs = filter(lambda x: f"csv" in x and MODELO in x,
                    os.listdir(os.getcwd()))
lista_csvs = list(lista_csvs)
if path_df_final in lista_csvs:
    lista_csvs.remove(path_df_final)
if path_df_resumo in lista_csvs:
    lista_csvs.remove(path_df_resumo)


df_final = None
fold = 1
df_resumo = []
for path in lista_csvs:
    df = pd.read_csv(path, index_col=0, decimal=",")
    tloss = df.loc[:, "train_loss"].mean()
    mae = df.loc[:, "MAE"].mean()
    dp = df.loc[:, "DP"].mean()
    pr = df.loc[:, "Precision"].mean()
    re = df.loc[:, "Recall"].mean()
    f1 = df.loc[:, "F1-score"].mean()
    t = df.loc[:, "time"].mean()

    df_resumo.append({
        "fold": f"FOLD_{fold}",
        "train_loss": tloss,
        "MAE": mae,
        "DP": dp,
        "Precision": pr,
        "Recall": re,
        "F1-score": f1,
        "time": t,
    })

    indices_novos = [f"FOLD_{fold}_{epoca}" for epoca in df.index.tolist()]
    df.index = indices_novos
    print(df.head(5), "\n\n")

    if FINAL:
        if df_final is None:
            df_final = df
        else:
            df_final = pd.concat([df_final, df])

    fold += 1

if FINAL:
    df_final.to_csv(path_df_final, decimal=",")

df_resumo = pd.DataFrame(df_resumo)
df_resumo.index = df_resumo.loc[:, "fold"]
df_resumo.to_csv(path_df_resumo, decimal=",")
