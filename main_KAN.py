import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, Dataset

import pandas as pd
import os
import sys
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

indexer = pd.IndexSlice


# ideia principal desse modelo é fazer deteccao da bounding box com as camadas convolucionais e depois calcular estrabismo
# com regressão na KAN
class KAN_CNN(nn.Module):
    def __init__(self, layers_hidden, polynomial_order=2, base_activation=nn.ReLU):
        super(KAN_CNN, self).__init__()

        # quant de hidden layers
        self.layers_hidden = layers_hidden
        self.polynomial_order = polynomial_order
        # funcao de ativaçao base pra ativar as camadas de input
        self.base_activation = base_activation()

        # Feature extractor with Convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3, stride=1, padding=1
            ),  # 1 input channel (grayscale), 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Calculate the flattened feature size after convolutional layers
        flat_features = 32 * 7 * 7
        self.layers_hidden = [flat_features] + self.layers_hidden

        self.base_weights = nn.ModuleList()
        self.poly_weights = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for in_features, out_features in zip(
            self.layers_hidden[:-1], self.layers_hidden[1:]
        ):
            self.base_weights.append(nn.Linear(in_features, out_features))
            self.poly_weights.append(
                nn.Linear(in_features * (polynomial_order + 1), out_features)
            )
            self.batch_norms.append(nn.BatchNorm1d(out_features))

    # funcao auxiliar pra computer monomios rapidamente(muito usado na KAN)
    def compute_efficient_monomials(self, x, order):
        powers = torch.arange(order + 1, device=x.device, dtype=x.dtype)
        x_expanded = x.unsqueeze(-1).repeat(1, 1, order + 1)
        return torch.pow(x_expanded, powers)

    def forward(self, x):
        # Reshape input from [batch_size, 784] to [batch_size, 1, 28, 28] for MNIST

        # nao precisa pra minha rede
        x = x.view(-1, 1, 28, 28)

        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the features from the conv layers

        for base_weight, poly_weight, batch_norm in zip(
            self.base_weights, self.poly_weights, self.batch_norms
        ):
            base_output = base_weight(x)
            monomial_basis = self.compute_efficient_monomials(x, self.polynomial_order)
            monomial_basis = monomial_basis.view(x.size(0), -1)
            poly_output = poly_weight(monomial_basis)
            x = self.base_activation(batch_norm(base_output + poly_output))

        return x


class ConvModule(nn.Module):
    def __init__(self, layers_hidden, base_activation=nn.ReLU):
        super(ConvModule, self).__init__()

        # quant de hidden layers
        self.layers_hidden = layers_hidden
        # Feature extractor with Convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3, stride=1, padding=1
            ),  # 1 input channel (grayscale), 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.base_activation = base_activation

    def forward(self, x):
        # Reshape input from [batch_size, 784] to [batch_size, 1, 28, 28] for MNIST

        # nao precisa pra minha rede
        x = self.feature_extractor(x)
        x = self.base_activation(x)
        return x


class Trainer:
    def __init__(self, PATH_YOLO, epochs=10):

        dataset = MyDataset(PATH_YOLO)
        DATALOADER = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
        model = ConvModule()
        criterion = BboxLoss()  # For classification tasks

        optimizer = torch.optim.Adam(model.parameters())
        for epoch in range(epochs):  # Define the number of epochs
            for inputs, labels in DATALOADER:  # Assuming data_loader is defined
                # Forward pass
                outputs = model(inputs)
                print(type(outputs), outputs)
                pred_bboxes = outputs
                target_bboxes = labels
                loss = criterion(pred_bboxes, target_bboxes, target_scores_sum, fg_mask)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self):
        """Initialize the BboxLoss module"""
        super().__init__()

    def forward(
        self,
        pred_bboxes,
        target_bboxes,
        target_scores_sum,
        fg_mask,
    ):
        """IoU loss."""
        # weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = Metricas.bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, GIoU=True
        )
        loss_iou = ((1.0 - iou)).sum() / target_scores_sum

        return loss_iou


class Metricas:
    def __init__(self):
        pass

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
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
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
    def __init__(self, PATH_YOLO):
        # Store the data and labels
        self.inicializar_dataset(PATH_YOLO)
        df_dados = self.df
        data = list(df_dados.loc[:, "PATH"].to_dict().values())
        self.data = np.array(data)
        # print("DATA: ", data)
        labels = list(df_dados.loc[:, "LABEL"].to_dict().values())
        self.labels = np.array(labels)
        assert len(self.labels) == len(self.data)

    def __len__(self):
        # Total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the data and label at the given index
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

    def __getitems__(self):
        return self.data, self.labels

    def generate_data(self, PATH_YOLO):
        """
        Faz walk no path especificado e retorna lista de imagens e labels
        """

        lista_imagens = list()
        lista_labels = list()
        lista_dirs = ["train", "valid", "test"]
        for diretorio in lista_dirs:
            print("DIRETORIO: ", diretorio)
            print("_____________________________________")
            images = os.path.join(PATH_YOLO, diretorio, "images")
            labels = os.path.join(PATH_YOLO, diretorio, "labels")
            for root, dir, lista_arq in os.walk(images):
                print(root)
                for arq_imagem in lista_arq:
                    if ".JPG" not in arq_imagem:
                        continue
                    arq = os.path.basename(arq_imagem).replace(".JPG", "")
                    arq_imagem = os.path.join(images, arq_imagem)
                    arq_label = os.path.join(
                        labels, os.path.basename(arq_imagem).replace(".JPG", ".txt")
                    )
                    lista_imagens.append(arq_imagem)
                    lista_labels.append(arq_label)

        return lista_imagens, lista_labels

    def inicializar_dataset(self, PATH_YOLO):
        """
        Gera df escreve em csv contendo o df
        Retorna IDs de Imagem, labels de imagem
        """
        POSICOES = ["PPO", "INFRA", "SUPRA", "LEVO", "DEXTRO"]

        lista_img, lista_labels = self.generate_data(PATH_YOLO)
        lista_img = sorted(lista_img, key=lambda x: os.path.basename(x).split("-")[0])
        lista_labels = sorted(
            lista_labels, key=lambda x: os.path.basename(x).split("-")[0]
        )
        # ordenacao necessaria pra garantir repoducibilidade
        tuplas_info = []
        for x in lista_img:
            splitado = os.path.basename(x).replace(".JPG", "").split("-")
            if splitado[1] not in POSICOES or len(splitado) != 2:
                continue
            id = splitado[0]
            pos = splitado[1]
            tuplas_info.append((id, pos))

        dict_df = {"PATH": None, "LABEL": None}
        multi_index = pd.MultiIndex.from_tuples(tuplas_info, names=["ID", "POSICAO"])
        self.df = pd.DataFrame(dict_df, index=multi_index)

        # povoa o dataframe com path de imagem e de label para cada paciente e posicao
        for path_img, path_label in zip(lista_img, lista_labels):
            splitado = os.path.basename(path_img).replace(".JPG", "").split("-")
            ID, POSICAO = splitado[:2]
            if POSICAO not in POSICOES:
                continue
            self.df.loc[indexer[ID, POSICAO], ["PATH", "LABEL"]] = [
                path_img,
                path_label,
            ]

        self.X = np.array(self.df.index.to_list())
        self.y = np.array(self.df["LABEL"].tolist())

        # utiliza id de paciente como id para grupo
        self.grupos_pac = self.df.index.get_level_values(0).tolist()

        """path_out = "out.csv"
        if os.path.exists(path_out):
            os.remove(path_out)
        self.df.to_csv(path_out, index=True)"""

        """ 
        for tupla, label in zip(X, y):
            id = tupla[0]
            grupos[id].append(label) """
        return
