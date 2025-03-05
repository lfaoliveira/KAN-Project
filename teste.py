import os
import pandas as pd
modelo = "MLP"
lista_csvs = filter(lambda x: f"csv" in x and modelo in x,
                    os.listdir(os.getcwd()))
lista_csvs = list(lista_csvs)

df_final = None
fold = 1
for path in lista_csvs:
    df = pd.read_csv(path)
    if df_final is None:
        df_final = df
    else:
        df_final = pd.concat([df_final, df])
# TODO: BOTAR NOS INDICES LOGICA DOS FOLDS

df_final.to_csv("FINAL.csv")
