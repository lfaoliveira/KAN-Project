from multiprocessing import freeze_support
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torch.optim as optim
from torchvision.io import decode_image
from torchvision.transforms.functional import resize
import pandas as pd
import os
import gc
import shutil
from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection import train_test_split
from kan import KAN


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

indexer = pd.IndexSlice
RAND_STATE_GERAL = 42
seed_pesos = 3
torch.manual_seed(seed_pesos)
# ideia principal desse modelo é fazer deteccao da bounding box com as camadas convolucionais e depois calcular estrabismo
# com regressão na KAN


class Conv_KAN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.INPUT_MLP = 100
        # 2 numeros para a MLP
        self.OUTPUT_MLP = 2
        # Define the backbone CNN
        channel_out = [16, 32, 64]

        self.model = nn.Sequential(
            # parte convolucional
            nn.Conv2d(3, channel_out[0], kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(channel_out[0], channel_out[1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel_out[1], channel_out[2],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            KAN(width=[10, 10, 2], grid=12, k=5, symbolic_enabled=False),
            KAN(width=[10, 10, 2], grid=12, k=5, symbolic_enabled=False)
        )
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)


""" class BboxLoss(nn.Module):
    '''Criterion class for computing training losses during training. Uses GIoU as a main loss'''

    def __init__(self):
        '''Initialize the BboxLoss module'''
        super().__init__()

    def forward(
        self,
        class_logits,
        bbox_preds,
        labels,
        bbox_targets,
    ):
        '''IoU loss.
        # weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = Metricas.bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, GIoU=True
        )
        loss_iou = ((1.0 - iou)).sum() / target_scores_sum
        '''
        classification_loss = F.cross_entropy(class_logits, labels)
        bbox_loss = F.smooth_l1_loss(bbox_preds, bbox_targets)
        return classification_loss + bbox_loss
 """


class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        self.INPUT_MLP = 100
        # 2 numeros para a MLP
        self.OUTPUT_MLP = 2
        # Define the backbone CNN
        channel_out = [16, 32, 64]
        self.model = nn.Sequential(
            # parte convolucional
            nn.Conv2d(3, channel_out[0], kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(channel_out[0], channel_out[1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel_out[1], channel_out[2],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            # parte MLP
            nn.LazyLinear(out_features=self.INPUT_MLP),
            nn.ReLU(),
            nn.LazyLinear(out_features=self.OUTPUT_MLP),
        )
        self.model = torch.compile(self.model)

        """ # Define the classifier head
        self.classifier = nn.Sequential(
            nn.Linear(out_features=num_classes),
            # Add more layers as needed
        )
        # Define the bounding box regressor head
        self.bbox_regressor = nn.Sequential(
            nn.Linear(out_features=4),
            # Add more layers as needed
        ) """

    def forward(self, x):
        out = self.model(x)
        # Flatten features for the heads
        # features = features.view(features.size(0), -1)
        # class_logits = self.classifier(in_features=features)
        # bbox_coords = self.bbox_regressor(in_features=features)
        return out


class Trainer:
    def __init__(self, PATH_YOLO, filename_tabela):
        """
        Parameters\n
        ---------------
        ``PATH_YOLO``: path containing images and labels in the YOLO format.
        ``epochs``:
        ``decay_step``: how many epochs to decay lr
        ``lr_decay``: factor for subtracting lr.
        """
        try:
            self.device = torch.cuda.current_device()
            print(f"DEVICE: {self.device}\n\n")
        except:
            self.device = torch.device("cpu")
            print(f"DEVICE: CPU\n\n")

        self.dataset = MyDataset(PATH_YOLO, filename_tabela, self.device)
        self.X = self.dataset.data.detach().cpu()
        self.y = self.dataset.labels.detach().cpu()
        self.quant_folds = 5

    def train(self, epochs=10, decay_step=7, lr_decay=1e-5, early_stop=10, warmup=5):

        # fn_loss = BboxLoss()  # For classification tasks
        fn_loss = nn.MSELoss()
        # variaveis auxiliares
        threshold = 10
        best_f1 = 0
        nochange = 0
        # num_workers = min(32, os.cpu_count() // 2)     # usar so quando nao bugar
        num_workers = 0
        results = []

        kfold = GroupKFold(
            n_splits=self.quant_folds, shuffle=True, random_state=RAND_STATE_GERAL
        )

        self.batch_size = 8
        num_exp = 0

        for train_idx, val_idx in kfold.split(self.X, self.y, groups=self.dataset.grupos_pac):
            print("-------------------------------")
            print(f"EXPERIMENTO N° {num_exp}")
            print("-------------------------------\n")

            model = Conv_KAN().to(self.device)
            # dry run pra inicializar LazyModules
            # shape: (Batch, Canais, Height, Width)
            model(torch.ones(size=(1, 3, 512, 512)).to(self.device))
            """ NOTE!!!!!!!!!!!!!!: Treinando modelo atualmente sem fazer transfer learning proprieamente dito """

            optimizer = optim.Adam(
                model.parameters(), weight_decay=1e-2, betas=(0.9, 0.99), lr=1e-2)
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=decay_step, gamma=lr_decay
            )

            best_f1 = 0
            nochange = 0
            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)

            self.TRAIN_LOADER = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
            )
            self.VAL_LOADER = DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, drop_last=True
            )

            for epoch in range(epochs):  # Define the number of epochs
                model.train()
                t1 = time.time()
                loss_epoch = []
                for image_batch, targets in self.TRAIN_LOADER:
                    # convertion to GPU tensor
                    image_batch = image_batch.to(self.device)
                    label_batch = targets.to(self.device)

                    # Forward pass
                    optimizer.zero_grad()
                    estrabismo = model(image_batch)
                    loss = fn_loss(estrabismo, label_batch)
                    loss_epoch.append(loss.detach())

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                loss_epoch = torch.stack(loss_epoch, dim=0)
                loss_mean = loss_epoch.mean().cpu().item()
                string_res = f"Epoch {epoch} of {epochs}, \
                    LOSS(MSE): {round(loss_mean, 3)},  \
                    Time: {round(time.time()-t1, 2)} seconds"

                print(string_res)
                lr_scheduler.step()

                # Validation loop
                model.eval()
                y_true = []
                y_pred = []
                with torch.no_grad():
                    for batch_X, batch_y in self.VAL_LOADER:
                        y_true.append(batch_y)
                        outputs = model(batch_X)
                        y_pred.append(outputs)

                    y_true = torch.stack(y_true, dim=0)
                    y_pred = torch.stack(y_pred, dim=0)
                    val_prec, val_rec, val_f1, MAE, DP = Metricas.metricas_val(
                        y_true, y_pred, threshold)

                    res = f"MAE: {round(MAE, 1)} DP: {round(DP, 1)} PREC: {val_prec} REC: {val_rec} F1: {val_f1}\n"
                    print(res)

                    results.append({
                        "experiment": num_exp,
                        "epoch": epoch,
                        "loss": round(loss_mean, 3),
                        "MAE": round(MAE, 1),
                        "DP": round(DP, 1),
                        "Precision": val_prec,
                        "Recall": val_rec,
                        "F1-score": val_f1,
                        "time": round(time.time() - t1, 2)
                    })

                    if val_f1 != torch.nan and val_f1 > best_f1:
                        best_f1 = val_f1
                        nochange = 0
                    else:
                        nochange += 1
                        if nochange > early_stop + warmup:
                            break
            num_exp += 1
            gc.collect()

        results_df = pd.DataFrame(results)
        # results_df.to_csv("training_results.csv", index=False)  # Save to CSV
        print("Results:", results_df)
        return


class Metricas:
    def __init__(self):
        pass

    @staticmethod
    def metricas_val(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float):
        abs_error = torch.abs(torch.sub(y_true, y_pred))
        MAE = torch.mean(abs_error)
        DP_ERRO = torch.std(abs_error)
        TP = torch.count_nonzero(torch.where(abs_error <= threshold, 1, 0))
        # print(TP)
        # FP = torch.count_nonzero(torch.where(abs_error > threshold, 0, 1))
        FP = torch.tensor(abs_error.nelement()) - TP
        # print(FP)
        # FN = FP  In regression, FP and FN are equivalent in this context
        add = torch.add(TP, FP)
        # print(add)
        if add > 0:
            precision = torch.divide(TP, add)
        else:
            precision = torch.tensor(0)

        if add > 0:
            recall = torch.divide(TP, add)
        else:
            recall = torch.tensor(0)
        del add

        f1 = torch.divide(torch.multiply(
            2*precision, recall), torch.add(precision, recall))

        precision = precision.detach().cpu().item()
        recall = recall.detach().cpu().item()
        f1 = f1.detach().cpu().item()
        MAE = MAE.detach().cpu().item()
        DP_ERRO = DP_ERRO.detach().cpu().item()

        return precision, recall, f1, MAE, DP_ERRO

    @staticmethod
    def bbox_iou(box1, box2, xywh=True, GIoU=False, eps=1e-7):
        """
        Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
        Args:
            box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
            box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
            xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                                (x1, y1, x2, y2) format. Defaults to True.
            GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
            DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
            CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
        """
        # Get the coordinates of bounding boxes
        if xywh:  # transform from xywh to xyxy
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(
                4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        ).clamp_(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + eps
        # IoU
        iou = inter / union
        if GIoU:
            # convex (smallest enclosing box) width
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
            # convex height
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area
        return iou  # IoU


class MyDataset(Dataset):
    def __init__(self, PATH_YOLO, filename_tabela, device):
        # Store the data and labels
        self.POSICOES = ["PPO", "INFRA", "SUPRA", "LEVO", "DEXTRO"]
        self.path_tabela = os.path.join(PATH_DATASET, filename_tabela)
        self.inicializar_dataset(PATH_YOLO)
        # TODO: criar logica de treino e validação com split de 70/30
        df_dados = self.df
        lista_path_img = list(df_dados.loc[:, "PATH"].to_dict().values())
        data = np.array([resize(decode_image(path), (512, 512))
                        for path in lista_path_img], dtype=np.float32)

        # array com imagens
        self.data = torch.tensor(data, device=device, dtype=torch.float32)
        print("DATA: ", self.data.shape)
        # lista contendo tuplas de estrabismo H e V
        print("LABEL DF: ", self.df.loc[:, "LABEL"])
        labels = df_dados.loc[:, "LABEL"].to_list()
        # array com classes e bboxes
        # print(np.array(labels), np.array(labels).shape)
        self.labels = torch.tensor(labels, dtype=torch.float32, device=device)
        print("LABELS: ", self.labels.shape)
        assert len(self.labels) == len(self.data)

    def __len__(self):
        # Total number of samples
        return len(self.data)

    """def __getitem__(self, idx):
        # Fetch the data and label at the given index
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label """

    def __getitem__(self, idx):
        """
        Retorna imagem(tensor torch) e label(tensor torch)
        """
        image = self.data[idx]
        label = self.labels[idx]

        return image, label

    def generate_data(self, PATH_YOLO):
        """
        Faz walk no path especificado e retorna lista de imagens e labels
        """
        lista_imagens = list()
        lista_labels = list()

        dict_df = {}
        for posicao in self.POSICOES:
            ler_colunas = ['PAC.', 'TIPO', 'DH', 'TIPO.1', 'DV', 'FIXADOR']
            df = pd.read_excel(
                self.path_tabela, sheet_name=posicao, usecols=ler_colunas, index_col=0)
            df = df.rename(columns={"TIPO": "TIPO_H", "TIPO.1": "TIPO_V"})
            df.index = df.index.astype('<U16')
            dict_df[posicao] = df.dropna(how="any")

            # print(dict_df[posicao])
        print("--------------------------------------\n\n")

        lista_dirs = ["train", "valid", "test"]
        for diretorio in lista_dirs:
            print("DIRETORIO: ", diretorio)
            print("_____________________________________")
            images = os.path.join(PATH_YOLO, diretorio, "images")

            for root, dir, lista_arq in os.walk(images):
                print(root)
                for arq_imagem in lista_arq:
                    if ".JPG" not in arq_imagem:
                        continue

                    arq = os.path.basename(
                        arq_imagem).replace(".JPG", "")

                    splitado = os.path.basename(arq).split("-")
                    id = splitado[0]
                    pos = splitado[1]
                    if pos not in self.POSICOES:
                        continue

                    arq_imagem = os.path.join(images, arq_imagem)
                    id_format = str(int(id))

                    if id_format not in dict_df[pos].index.to_list():
                        continue
                    entrada_df = dict_df[pos].loc[id_format]
                    tupla_label = [f"{id}-{pos}",
                                   float(entrada_df["DH"]),
                                   float(entrada_df["DV"])]

                    lista_imagens.append(arq_imagem)
                    lista_labels.append(tupla_label)

        return lista_imagens, lista_labels

    def inicializar_dataset(self, PATH_YOLO):
        """
        Gera df escreve em csv contendo o df
        Retorna IDs de Imagem, labels de imagem
        """

        lista_img, lista_labels = self.generate_data(PATH_YOLO)
        # ordenacao necessaria pra garantir repoducibilidade
        lista_img = sorted(
            lista_img, key=lambda x: os.path.basename(x).split("-")[0])
        lista_labels = sorted(
            lista_labels, key=lambda x: x[0]
        )

        # cria tuplas pra inserir dentro do df
        tuplas_info = []
        for x in lista_img:
            splitado = os.path.basename(x).replace(".JPG", "").split("-")
            if splitado[1] not in self.POSICOES or len(splitado) != 2:
                continue
            id = splitado[0]
            pos = splitado[1]
            tuplas_info.append((id, pos))

        dict_df = {"PATH": None, "LABEL": None}
        multi_index = pd.MultiIndex.from_tuples(
            tuplas_info, names=["ID", "POSICAO"])
        self.df = pd.DataFrame(dict_df, index=multi_index)

        # povoa o dataframe com path de imagem e de label para cada paciente e posicao
        for path_img, tupla_label in zip(lista_img, lista_labels):
            splitado = os.path.basename(path_img) \
                .replace(".JPG", "").split("-")
            ID, POSICAO = splitado[:2]
            if POSICAO not in self.POSICOES:
                continue
            self.df.loc[indexer[ID, POSICAO], ["PATH", "LABEL"]] = [
                path_img,
                tupla_label[1:],
            ]
        """ self.X = np.array(self.df.index.to_list())
        self.y = np.array(self.df["LABEL"].tolist()) """

        # utiliza id de paciente como id para grupo
        # apenas usar em cross-validation.
        self.grupos_pac = self.df.index.get_level_values(0).tolist()

        return


# -------------------------------------MAIN------------------------------------#
# MODE LOCAL == running outside of Google Colab
MODO = "LOCAL"
if os.path.exists("/content"):
    MODO = "COLAB"
else:
    MODO = "LOCAL"
# --SETTING PATH_DATASET-- #
if MODO == "COLAB":
    PATH_DATASET = os.path.join("/content", "datasets")
else:
    PATH_DATASET = os.path.join(os.getcwd(), "datasets")


if MODO != "LOCAL":
    from google.colab.patches import cv2_imshow
    from google.colab import drive

    drive.mount("/content/drive")
    if not os.path.exists(os.path.join(PATH_DATASET, "SCRIPTS/")):
        shutil.copytree(
            "/content/drive/MyDrive/DATASETS DE SEGMENTAÇÃO/SCRIPTS/",
            os.path.join(PATH_DATASET, "SCRIPTS/")
        )
    if not os.path.exists(os.path.join(PATH_DATASET, "YOLO/")):
        shutil.copytree(
            "/content/drive/MyDrive/DATASETS DE SEGMENTAÇÃO/DADOS/YOLO",
            os.path.join(PATH_DATASET, "YOLO/")
        )
    # raise error when dataset not present
elif not os.path.exists(PATH_DATASET):
    raise SystemError("Dataset not found")


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    PATH_YOLO = os.path.join(PATH_DATASET, "YOLO")
    filename_tabela = "DiagnosticoEspecialista_Tese_Dallyson (ATUALIZADO).xlsx"
    path_tabela = os.path.join(PATH_DATASET, filename_tabela)
    trainer = Trainer(PATH_YOLO, filename_tabela)
    freeze_support()
    trainer.train(epochs=100, early_stop=10)
    # trainer.train(epochs=10)
