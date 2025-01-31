import pstats
import cProfile
from multiprocessing import freeze_support
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, Dataset
import torch.optim as optim
from torchvision.io import decode_image
from torchvision.transforms.functional import resize
import pandas as pd
import os
import gc
import shutil

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
            monomial_basis = self.compute_efficient_monomials(
                x, self.polynomial_order)
            monomial_basis = monomial_basis.view(x.size(0), -1)
            poly_output = poly_weight(monomial_basis)
            x = self.base_activation(batch_norm(base_output + poly_output))

        return x


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
        channel_out = [16, 32]
        self.model = nn.Sequential(
            # parte convolucional
            nn.Conv2d(3, channel_out[0], kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel_out[0], channel_out[1],
                      kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            # parte MLP
            nn.LazyLinear(out_features=self.INPUT_MLP),
            nn.LazyLinear(out_features=self.OUTPUT_MLP),
        )
        torch.compile(self.model)

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
        # print("DF MYDATASET: ", self.dataset.df)
        num_workers = min(32, os.cpu_count() // 2)
        self.DATALOADER = DataLoader(
            self.dataset, batch_size=4, shuffle=True, num_workers=0
        )

    def train(self, epochs=10, decay_step=7, lr_decay=0.1):
        model = ConvModule().to(self.device)
        # fn_loss = BboxLoss()  # For classification tasks
        fn_loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=lr_decay
        )

        for epoch in range(epochs):  # Define the number of epochs
            model.train()
            loss_epoch = 0
            t1 = time.time()
            for images, targets in self.DATALOADER:
                # convertion to GPU tensor
                images = images.to(self.device)
                labels = targets.to(self.device)
                # Forward pass
                optimizer.zero_grad()
                estrabismo = model(images)
                # print(estrabismo, type(estrabismo), labels, type(labels))
                loss = fn_loss(estrabismo, labels)
                # Backward pass
                loss.backward()
                loss_epoch = torch.mean(loss.detach())
                optimizer.step()
            string_res = f"Epoch {epoch} of {epochs}, LOSS(MSE): {loss_epoch.flatten().cpu().item()}, Time: {time.time() - t1} seconds"
            print(string_res)
            lr_scheduler.step()

        gc.collect()
        return


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
        print(np.array(labels), np.array(labels).shape)
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
        self.X = np.array(self.df.index.to_list())
        self.y = np.array(self.df["LABEL"].tolist())

        # utiliza id de paciente como id para grupo
        # apenas usar em cross-validation.
        # self.grupos_pac = self.df.index.get_level_values(0).tolist()

        """path_out = "out.csv"
        if os.path.exists(path_out):
            os.remove(path_out)
        self.df.to_csv(path_out, index=True)"""

        """
        for tupla, label in zip(X, y):
            id = tupla[0]
            grupos[id].append(label) """
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
    print(torch.backends.cudnn.version())

    PATH_YOLO = os.path.join(PATH_DATASET, "YOLO")
    filename_tabela = "DiagnosticoEspecialista_Tese_Dallyson (ATUALIZADO).xlsx"
    path_tabela = os.path.join(PATH_DATASET, filename_tabela)
    trainer = Trainer(PATH_YOLO, filename_tabela)
    freeze_support()
    profiler = cProfile.Profile()
    profiler.enable()
    trainer.train(epochs=15)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("tottime").print_stats(15)
    # trainer.train(epochs=10)
