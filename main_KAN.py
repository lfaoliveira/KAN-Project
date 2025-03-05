from multiprocessing import freeze_support
import time
import numpy as np
from typing import Any, cast, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from torchvision.io import decode_image
import torchvision.models
from torchvision.transforms.functional import resize
import pandas as pd
import random
import os
import gc
import shutil
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import albumentations
from kan import KAN


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Ensure deterministic behavior for CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

indexer = pd.IndexSlice
RAND_STATE_GERAL = 42
torch.manual_seed(RAND_STATE_GERAL)
torch.cuda.manual_seed(RAND_STATE_GERAL)
torch.cuda.manual_seed_all(RAND_STATE_GERAL)
random.seed(RAND_STATE_GERAL)
np.random.seed(RAND_STATE_GERAL)

# ideia principal desse modelo é fazer deteccao da bounding box com as camadas convolucionais e depois calcular estrabismo
# com regressão na KAN


class Conv_KAN(nn.Module):
    def __init__(self, plot_ativ=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.INPUT_MLP = 512
        # 2 numeros para a MLP
        self.OUTPUT_MLP = 2
        # Define the backbone CNN
        cfg = [3, 64, 64, "M", 128, 128, "M", 256, 256,
               256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

        self.INPUT_MLP = 4096
        self.conv = self.setup_conv(cfg)

        self.model = nn.Sequential(
            self.conv,
            nn.Flatten(),
            KAN(width=[self.INPUT_MLP, 20, 20, 20, self.OUTPUT_MLP], grid=12, k=5,
                symbolic_enabled=False, seed=RAND_STATE_GERAL),
        )
        if plot_ativ:
            self.activations = {}
            self.hook_handle = self._modules.get("conv").register_forward_hook(
                self.save_activation("conv"))
            # self.model = torch.compile(self.model)

    def setup_conv(self, cfg: List[Union[str, int]]) -> nn.Sequential:
        # so confia que funciona
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                v = int(v)
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=3, stride=1, padding=1)
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook


class ConvModule(nn.Module):
    def __init__(self, plot_ativ=False):
        super(ConvModule, self).__init__()

        # 2 numeros para a MLP
        self.OUTPUT_MLP = 2
        # Define the backbone CNN
        cfg = [3, 64, 64, "M", 128, 128, "M", 256, 256,
               256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

        self.INPUT_MLP = 4096

        # parte convolucional
        self.conv = self.setup_conv(cfg)

        self.model = nn.Sequential(
            self.conv,
            nn.Flatten(),
            nn.LazyLinear(out_features=self.INPUT_MLP),
            nn.ReLU(),
            nn.LazyLinear(out_features=self.INPUT_MLP),
            nn.ReLU(),
            # strab horizontal e vertical
            nn.LazyLinear(out_features=2),
            nn.ReLU(),
        )
        a = torchvision.models.vgg16()
        if plot_ativ:
            self.activations = {}
            self.hook_handle = self._modules.get("conv").register_forward_hook(
                self.save_activation("conv"))
        else:
            self.model = torch.compile(self.model)

    def setup_conv(self, cfg: List[Union[str, int]]) -> nn.Sequential:
        # so confia que funciona
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                v = int(v)
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=3, stride=1, padding=1)
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

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

    def train(self, epochs=100, df_resumo=None, decay_step=20, lr_decay=0.8,
              early_stop=150, warmup=5, plot_ativ=False) -> pd.DataFrame:
        """
        Trains the model using k-fold cross validation

        Parameters:
        -----------
        epochs : int
            Number of training epochs
        df_resumo : pd.DataFrame, optional
            DataFrame to store training results across folds
        decay_step : int 
            Steps between learning rate decay
        lr_decay : float
            Learning rate decay factor
        early_stop : int
            Number of epochs without improvement before stopping
        warmup : int 
            Number of warmup epochs
        plot_ativ : bool
            Whether to plot activations

        Returns:
        --------
        pd.DataFrame
            DataFrame containing training results
        """
        fn_loss = nn.L1Loss()
        # variaveis auxiliares
        self.threshold = 10

        # num_workers = min(32, os.cpu_count() // 2)     # usar so quando nao bugar
        num_workers = 0

        # Setup k-fold cross validation
        kfold = GroupKFold(
            n_splits=self.quant_folds, shuffle=True, random_state=RAND_STATE_GERAL
        )

        self.batch_size = 4
        num_exp = 1
        model_str = ""
        # Iterate through folds
        for train_idx, val_idx in kfold.split(self.X, self.y, groups=self.dataset.grupos_pac):
            print("-------------------------------")
            print(f"EXPERIMENTO N° {num_exp}")
            print("-------------------------------\n")

            results = []
            train_losses = []
            eval_losses = []

            # Initialize model
            model = ConvModule(plot_ativ).to(self.device)

            # Determine model type for logging

            if isinstance(model, Conv_KAN):
                model_str = "KAN"
            else:
                model_str = "MLP"

            #  run pra inicializar LazyModules
            # shape: (Batch, Canais, Height, Width)
            model(torch.ones(size=(1, 3, 512, 512)).to(self.device))
            print("MODELO COMPILADO")
            """ NOTE!!!!!!!!!!!!!!: Treinando modelo atualmente sem fazer transfer learning proprieamente dito """

            # Setup optimizer and learning rate scheduler
            optimizer = optim.Adam(
                model.parameters(), weight_decay=1e-2, betas=(0.9, 0.999), lr=1e-2)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, eta_min=1e-10, T_0=warmup
            )

            best_loss = 0
            nochange = 0

            # Create train/val datasets for this fold
            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)
            path_imgs = np.array(self.val_dataset.dataset.lista_path_img)
            names = [
                os.path.basename(elem) for elem in path_imgs[val_idx]]

            # Create data loaders
            self.TRAIN_LOADER = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
            )
            self.VAL_LOADER = DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, drop_last=True
            )
            print("SETUP PRONTO")

            # Training loop
            for epoch in range(epochs):  # Define the number of epochs
                model.train()
                t1 = time.time()
                loss_epoch = []

                # Iterate through batches
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

                # Calculate epoch loss
                loss_epoch = torch.stack(loss_epoch, dim=0)
                loss_batch = loss_epoch.mean().numpy(force=True).item()
                if epoch + 1 > warmup:
                    if loss_batch < 100:
                        train_losses.append(float(loss_batch))
                    else:
                        train_losses.append(None)
                temp_batch = round(time.time() - t1, 2)
                string_res = f"Epoch {epoch} of {epochs}, \
                    LOSS(MAE): {round(loss_batch, 3)},  \
                    Time: {temp_batch} seconds"

                print(string_res)
                lr_scheduler.step()

                # Evaluate model
                kwargs = {
                    "epoch": epoch, "loss_batch": float(loss_batch), "t1": t1}
                val_f1, val_loss = self.avaliar(
                    model, results, fn_loss, **kwargs)
                eval_losses.append(val_loss)

                # Early stopping check
                if val_loss != None and val_loss != torch.nan and val_loss < best_loss:
                    best_loss = val_loss
                    nochange = 0
                else:
                    nochange += 1
                    if nochange > early_stop + warmup:
                        break
            if loss_batch < 500:
                train_losses.append(float(loss_batch))

            # Final evaluation and plotting
            self.eval_modelo(
                model, names, num_exp, plot_ativ)
            # Plot the loss
            plt.plot(train_losses, color='r', label='Train Loss')
            plt.plot(eval_losses, label='Val Loss', color='blue')
            plt.xlabel('Epoch')
            # Set y-axis to start at zero (to not have varying scales)
            plt.ylim(bottom=0)
            plt.ylabel('Loss')
            plt.title('Loss Over Epochs')
            plt.legend()
            plt.show()
            # Remove the hook to save the model
            if plot_ativ:
                model.hook_handle.remove()
            else:
                torch.save(model, f"KAN_{num_exp}.pt")

            # Cleanup and save results
            num_exp += 1
            del model
            gc.collect()
            torch.cuda.empty_cache()

            results_df = pd.DataFrame(results)
            indices_novos = [
                f"FOLD_{num_exp}_{str(epoca)}" for epoca in results_df.index.tolist()]
            results_df.index = indices_novos

            if df_resumo != None:
                df_resumo = pd.concat([df_resumo, results_df])
            else:
                df_resumo = results_df

            results_df.to_csv(f"training_results_{model_str}_{num_exp}.csv",
                              index=False, decimal=",")  # Save to CSV

        # print("Results:", results_df)
        return df_resumo, model_str

    def eval_modelo(self, model, names, num_exp, plot_ativ=False):
        """
        Fazendo eval final do ultimo modelo
        """
        model.eval()
        y_true = []
        y_pred = []
        imgs = []
        ativacoes = []
        with torch.no_grad():
            for batch_X, batch_y in self.VAL_LOADER:
                imgs.append(batch_X)
                y_true.append(batch_y)
                outputs = model(batch_X)
                act = model.activations['conv'].squeeze()
                ativacoes.append(act)
                y_pred.append(outputs)

            tensor_imgs = torch.cat(imgs, dim=0)
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)

            dir_save = os.path.join(os.getcwd(), f"EXP {num_exp}")
            os.makedirs(dir_save, exist_ok=True)
            plot_image_with_number(
                tensor_imgs, y_true, y_pred, names, save_dir=dir_save)
            if plot_ativ:
                ativacoes = torch.cat(ativacoes)
                # plotar feature maps
                plot_maps(ativacoes)

            return

    def avaliar(self, model, results, fn_loss, **kwargs):
        epoch = kwargs.get('epoch', None)
        loss_batch = kwargs.get('loss_batch', None)
        t1 = kwargs.get('t1', None)
        # Validation loop
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            media = []
            for batch_X, batch_y in self.VAL_LOADER:
                y_true.append(batch_y)
                outputs = model(batch_X)
                loss = fn_loss(outputs, batch_y)
                media.append(loss)
                y_pred.append(outputs)

            y_true = torch.stack(y_true, dim=0)
            y_pred = torch.stack(y_pred, dim=0)
            loss = torch.stack(media, dim=0).mean().item()
            val_prec, val_rec, val_f1, DP = Metricas.metricas_val(
                y_true, y_pred, self.threshold)

            res = f"LOSS_VAL: {loss} PREC: {val_prec} REC: {val_rec} F1: {val_f1}\n"
            print(res)
            temp_total = round(time.time() - t1, 2)
            results.append({
                "epoch": epoch,
                "train_loss": round(loss_batch, 3),
                "MAE": loss,
                "DP": DP,
                "Precision": val_prec,
                "Recall": val_rec,
                "F1-score": val_f1,
                "time": temp_total
            })
        if loss >= 500:
            loss = None
        return val_f1, loss


class Metricas:
    def __init__(self):
        pass

    @staticmethod
    def metricas_val(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float):
        abs_error = torch.abs(torch.sub(y_true, y_pred))
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

        precision = precision.detach().round(decimals=2).numpy(force=True).item()
        recall = recall.detach().round(decimals=2).numpy(force=True).item()
        f1 = f1.detach().round(decimals=2).numpy(force=True).item()
        DP_ERRO = DP_ERRO.detach().round(decimals=2).numpy(force=True).item()

        precision, recall, f1, DP_ERRO = round(precision, 2), round(
            recall, 2), round(f1, 2), round(DP_ERRO, 2)

        return precision, recall, f1, DP_ERRO

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
        df_dados = self.df
        self.lista_path_img: List[str] = list(
            df_dados.loc[:, "PATH"].to_dict().values())

        a = [resize(decode_image(path), [512, 512])
             for path in self.lista_path_img]
        data = np.array(a, dtype=np.float32)

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

    def __getitem__(self, idx):
        """
        Retorna imagem(tensor torch) e label(tensor torch)
        """
        image = self.data[idx]
        # Apply Albumentations transformations
        transform = albumentations.Compose([
            albumentations.ShiftScaleRotate(
                shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1), rotate_limit=(-10, 10), p=0.8),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.8),
            albumentations.HorizontalFlip(p=0.8),
        ], seed=int(idx))
        augmented = transform(
            image=image.detach().cpu().numpy().transpose(1, 2, 0) / 255.0)

        image = torch.tensor(augmented['image'].transpose(
            2, 0, 1), dtype=torch.float32, device=image.device)

        # Normalize the image for plotting
        img_plot = augmented['image']

        # Plot and show the transformed image
        """ 
        plt.figure()
        plt.imshow(img_plot)
        plt.title(f'Transformed Image {idx}')
        plt.axis('off')
        plt.show()
        plt.close() """

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


def plot_image_with_number(tensor_imgs: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, names, save_dir):
    """
    Displays an image with a number caption below it and saves it to disk.\n
    ------
    Parameters:
    - tensor_imgs: torch tensors containing one image each entry
    - y_true: torch tensors containing strabismus labels
    - y_pred: torch tensors containing strabismus predictions
    """

    # iterates idx over batch size
    for idx in range(tensor_imgs.shape[0]):
        img = tensor_imgs[idx].numpy(force=True).transpose(1, 2, 0) / 255
        label = y_true[idx].numpy(force=True)
        pred = y_pred[idx].numpy(force=True)

        path_out = f"{os.path.splitext(names[idx])[0]}_OUTPUT.jpg"
        # Create a figure and axis
        fig, ax = plt.subplots()
        # Display the image
        # ax.imshow(img)
        ax.axis('off')  # Hide the axes
        # Add the number below the image
        str_estrab = f"PRED:{pred}, LABEL:{label}"
        plt.figtext(0.5, 0.01, str_estrab, ha='center', fontsize=12)
        fig.savefig(os.path.join(save_dir, path_out))
        # Show the plot
        # print("\n")
        # plt.show()
        plt.close()


def plot_maps(ativacoes: torch.Tensor):
    # Select the batch to visualize
    act_map = ativacoes[0]

    # Filter out empty sub-tensors
    act_map = torch.stack([x for x in act_map if torch.any(x)])

    # Number of channels in the activation map
    num_channels = act_map.shape[0]

    # Determine grid size for plotting
    grid_size = int(num_channels ** 0.5)
    if grid_size ** 2 < num_channels:
        grid_size += 1

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
    fig.suptitle(f'Activation Maps for Layer: CONV', fontsize=16)

    # Plot each channel's activation map
    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        if i < num_channels:
            channel_map = act_map[i].cpu().numpy()
            ax.imshow(channel_map, cmap='viridis')
            ax.set_title(f'Channel {i}', fontsize=8)
        else:
            ax.axis('off')

        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


# -------------------------------------MAIN------------------------------------#
# MODE LOCAL == running outside of Google Colab
MODO = "LOCAL"
if os.path.exists("/content"):
    MODO = "COLAB"

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
    raise SystemError(
        "Dataset not found, please put it inside the working directory")


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    PATH_YOLO = os.path.join(PATH_DATASET, "YOLO")
    filename_tabela = "DiagnosticoEspecialista_Tese_Dallyson (ATUALIZADO).xlsx"
    path_tabela = os.path.join(PATH_DATASET, filename_tabela)
    trainer = Trainer(PATH_YOLO, filename_tabela)
    freeze_support()
    df_resumo, exp = trainer.train(epochs=100, early_stop=25, plot_ativ=True)
    df_resumo.to_csv(f"df_resumo_{exp}.csv")
