import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration
from skimage import io, filters, img_as_float
from skimage.filters import unsharp_mask
import copy
from torch.utils.data import Subset, Dataset
import torch
import yaml

"""import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2"""
import matplotlib.patches as patches
import cv2
from ultralytics import YOLO
import os
import shutil
import pandas as pd
import gc
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import sys

""" usar em caso de algum bug estranho de
locale.getpreferredencoding = lambda: "UTF-8" """

POSICOES = ["PPO", "DEXTRO", "LEVO", "SUPRA", "INFRA"]

# ---------------------------------------------------------------------------------------------#


# CLASSE DE TESTE DE ESTRABISMO
# PASSAR DATASET 'todo provavelmente
class Deteccao:
    def __init__(self, modelo_path, dict_data: dict):
        self.modelo_path = modelo_path
        self.modelo = YOLO(modelo_path).to("cuda")
        self.errados = 0  # imagens sem numero minimo de deteccao
        self.no_detect = 0  # para quando nao ha NENHUMA deteccao na imagem
        self.validos = 0  # imagens, do total, que foram realmente analisadas
        self.fora_tabela = 0  # imagem fora da tabela
        self.dict_data = dict_data
        self.altos = list()

    def loop_detect(
        self,
        PATH_DATASET,
        PATH_SAVE,
        PATH_DADOS,
        experimento,
        lista_dirs,
        auxiliar,
        metricas,
        manipulador_loop,
        check_SAM,
    ):

        n = 0
        self.errados = 0
        self.no_detect = 0
        self.validos = 0
        self.fora_tabela = 0

        # maximo de imagens a serem analisadas
        maximo = np.inf
        names = ["LIMBO", "FLASH"]
        estrabismo = Estrabismo()

        pac_removidos = list()
        SALVAR = True
        # TODO: GANRANTIR QUE DATASET ESTA SENDO ANALISADO CORRETAMENTE

        print("DIRETORIO: ", PATH_DADOS)
        print("_____________________________________")
        images = os.path.join(PATH_DADOS, "images")
        labels = os.path.join(PATH_DADOS, "labels")
        for root, dir, lista_arq in os.walk(images):
            print(root)
            for arq_imagem in lista_arq:
                if ".JPG" not in arq_imagem and ".jpg" not in arq_imagem:
                    continue
                arq_imagem = os.path.join(images, arq_imagem)
                arq_label = os.path.join(
                    labels, os.path.basename(
                        arq_imagem).replace(".JPG", ".txt")
                )
                arq = (
                    os.path.basename(arq_imagem).replace(
                        ".JPG", "").replace(".jpg", "")
                )
                imagem = cv2.imread(arq_imagem)
                height, width = imagem.shape[:2]
                n += 1
                print(arq)

                offset = width // 2
                lado_esq, lado_dir = imagem[:, :offset], imagem[:, offset:]

                dividido = os.path.basename(arq).split("-")
                num_paciente = dividido[0]
                POSICAO = dividido[1]
                if POSICAO not in POSICOES:
                    continue
                POSICAO = POSICAO.split("_")[0]

                for pos_lado, lado in enumerate([lado_esq, lado_dir]):
                    # funcao detectanolhos e brilho com a YOLO e guarda em "auxiliar" #
                    if not self.detect_objects(
                        arq_imagem, pos_lado, lado, auxiliar, offset
                    ):
                        print("DEU PROBLEMA NA DETECCAO")
                        continue

                copia = imagem.copy()
                for class_name in names:
                    i = 0
                    for array in auxiliar.caixas.get_dado(
                        num_paciente, [POSICAO, class_name.upper()]
                    ):
                        auxiliar.drawClasse(class_name, array, copia, i)
                        i += 1
                if SALVAR:
                    cv2.imwrite(os.path.join(
                        PATH_SAVE, arq + " N0" + ".jpg"), copia)

                manipulador_loop.inserir_dado_paciente(
                    num_paciente, [POSICAO, "ID"], num_paciente
                )
                data = self.dict_data[POSICAO]

                chaves = np.array(data.index, dtype=np.dtype(int))
                indice = chaves[chaves == int(num_paciente)]

                # Paciente fora da tabela
                if len(indice) == 0 or indice is None:
                    print("Paciente não esta na tabela\n")
                    # sempre usa label de indice
                    manipulador_loop.remover_paciente(num_paciente)
                    if arq not in pac_removidos:
                        pac_removidos.append(arq)
                    self.fora_tabela += 1
                    print("--------------------------------------------\n")
                    continue

                with open("QUANT_IMG_COM_ANOT.txt", "a") as f:
                    f.write(f"{arq}\n")

                indice = str(int(indice))

                quant_flash = len(
                    auxiliar.caixas.get_dado(
                        num_paciente, [POSICAO, "flash".upper()])
                )
                quant_limbo = len(
                    auxiliar.caixas.get_dado(
                        num_paciente, [POSICAO, "limbo".upper()])
                )
                if quant_flash < 2 or quant_limbo < 2:
                    print("QUANT DE LIMBOS OU BRILHOS INSUFICIENTE\n\n")
                    self.errados += 1
                    continue

                problema = False

                if problema:
                    print("QUANT DE LIMBOS OU BRILHOS INSUFICIENTE")
                    self.errados += 1
                    continue

                copia = imagem.copy()
                # desenha cada classe em uma copia da imagem principal
                for class_name in names:
                    i = 0
                    for array in auxiliar.caixas.get_dado(
                        num_paciente, [POSICAO, class_name.upper()]
                    ):
                        auxiliar.drawClasse(class_name, array, copia, i)
                        i += 1

                auxiliar.anotar_true(copia, arq_label)
                if SALVAR:
                    cv2.imwrite(os.path.join(
                        PATH_SAVE, arq + " N1" + ".jpg"), copia)

                # FUNCAO QUE PEGA DESVIOS ANOTADOS DOS PACIENTES
                # desvios_true: list[dh, dv]
                desvios_true = auxiliar.pega_desvios(
                    indice, POSICAO, printar=False)

                if desvios_true is False:
                    continue
                assert desvios_true != False

                # pegada de anotacoes dos olhos
                centros_limbo, flash_olhos = auxiliar.get_label(
                    arq_label, width, height
                )

                for centro in centros_limbo:
                    auxiliar.caixas_true.get_dado(
                        num_paciente, [POSICAO, "LIMBO".upper()]
                    ).append(centro)
                for centro in flash_olhos:
                    auxiliar.caixas_true.get_dado(
                        num_paciente, [POSICAO, "FLASH"]
                    ).append(centro)

                # criacao de metricas
                manipulador_loop.adicionar_metricas(
                    names, self.errados, POSICAO, auxiliar, num_paciente
                )

                # OBS: USANDO-SE DIAM DOS LIMBOS
                diam_limbos = list()
                for rect in auxiliar.caixas.get_dado(num_paciente, [POSICAO, "LIMBO"]):
                    # altura e largura da bbox
                    diam_limbos.append([rect[2], rect[3]])

                flashs = auxiliar.caixas.get_dado(
                    num_paciente, [POSICAO, "FLASH"])
                limbos = auxiliar.caixas.get_dado(
                    num_paciente, [POSICAO, "LIMBO"])
                printar = False
                # calculo do estrabismo
                dh, dv, pos_desv = estrabismo.estrabismo_flash(
                    flashs, limbos, imagem, diam_limbos, printar
                )
                copia = imagem.copy()
                print(f"DH: {dh} DV: {dv}")

                manipulador_loop.inserir_dado_paciente(
                    num_paciente, [POSICAO, "Desvio_Calc"], [dh, dv]
                )
                manipulador_loop.inserir_dado_paciente(
                    num_paciente, [POSICAO, "Desvio_True"], desvios_true
                )

                aux_h, aux_v = desvios_true[0], desvios_true[1]
                erro_h, erro_v = abs(aux_h - dh), abs(aux_v - dv)
                manipulador_loop.inserir_dado_paciente(
                    num_paciente, [POSICAO, "Erro_H"], erro_h
                )
                manipulador_loop.inserir_dado_paciente(
                    num_paciente, [POSICAO, "Erro_V"], erro_v
                )

                print(f"ERRO_H:{erro_h}  ERRO_V:{erro_v}")

                olho_true = data["FIXADOR"][indice][0]
                # troca referencia para esq e dir da imagem
                olho_true = "OE" if olho_true == "OD" else "OD"
                olho_pred = "OE" if pos_desv == 0 else "OD"
                # dict_para_df[POSICAO]['OLHO_CORRETO'][num_paciente] = olho_pred == olho_true
                print(
                    f"OLHO DESVIO TRUE: {olho_true}, OLHO DESVIO PRED:{olho_pred}")

                auxiliar.mostrar_errado(
                    imagem,
                    POSICAO,
                    self.altos,
                    arq_label,
                    num_paciente,
                    (erro_h, erro_v),
                )
                print(
                    f"MEDIDA_H: {desvios_true[0]} MEDIDA_V: {desvios_true[1]}")
                if SALVAR:
                    cv2.imwrite(os.path.join(
                        PATH_SAVE, arq + " N2" + ".jpg"), imagem)
                # p = os.path.join(IMG_SALVAS,f"Anotado {str(dir)} {str(posicao)}.png ")
                # cv2.imwrite(p, annotated_image)
                print("--------------------------------------------\n")
                # gc.collect()
                # limpar_cuda()
                if n >= maximo:
                    break

        print(f"TOTAL DE IMAGENS: {n}")
        self.validos = n - self.no_detect - self.fora_tabela

        print("N_IMAGENS ANALISADAS:", self.validos)
        print(
            f"ERRO DE DETECÇÃO:  {self.errados} = {round(self.errados / n * 100, 2):.2f} %"
        )
        print("N_IMAGENS VALIDAS:", self.validos - self.errados)
        print(
            f"PERC. ERROS > 10 DIOPTRIAS: {len(self.altos)/(self.validos)*100:.2f}%")
        return pac_removidos

    def detect_objects(self, arq_imagem, pos_lado, imagem, auxiliar, offset):
        height, width = imagem.shape[:2]
        resultado = self.modelo.predict(
            imagem,
            augment=True,
            device=0,
            save=False,
            iou=0.1,
            conf=0.1,
            imgsz=(height, width),
            max_det=5000,
        )
        if len(resultado[0]):
            arq = os.path.basename(arq_imagem).replace(
                ".jpg", "").replace(".JPG", "")
            dividido = os.path.basename(arq).split("-")
            num_paciente = dividido[0]
            POSICAO = dividido[1]
            POSICAO = POSICAO.split("_")[0]

            resultado = resultado[0]
            names = resultado.names
            BBOX = resultado.boxes  # bounding box
            boxes = BBOX.xywh.int()
            classes = BBOX.cls.int()
            scores = BBOX.conf  # Assuming 'scores' gives the confidence scores

            centers_x = (boxes[:, 0]).int()
            centers_y = (boxes[:, 1]).int()
            wids = (boxes[:, 2]).int()
            heis = (boxes[:, 3]).int()

            # Get class names
            class_names = [names[int(i)] for i in classes]

            """print(" ")
      print(f"CLASS_NAMES: {class_names}")
      print(scores)"""
            # cada paciente possui um dicionario com as classes
            # cada classe possui uma lista com sua bounding box direita e esquerda, em ordem
            limbo_lados = list()
            flash_lados = list()
            dict_cls_lados = {"limbo": limbo_lados, "flash": flash_lados}

            # Store centers of "limbo" and "flash" classes along with confidence scores
            for class_name, center_x, center_y, w, h, score in zip(
                class_names, centers_x, centers_y, wids, heis, scores
            ):
                if class_name in names.values():
                    # lado direito precisa ajustar para coordenadas absolutas
                    if pos_lado == 1:
                        center_x = center_x.cpu() + offset
                        pass
                    array = [
                        center_x.cpu().item(),
                        center_y.cpu().item(),
                        w.cpu().item(),
                        h.cpu().item(),
                        score.cpu().item(),
                    ]
                    dict_cls_lados[class_name].append(array)
            # Loop para eliminar limbos sem brilho
            limbo_lados = [
                limbo
                for limbo in dict_cls_lados["limbo"]
                if any(
                    auxiliar.in_caixa(limbo, brilho)
                    for brilho in dict_cls_lados["flash"]
                )
            ]
            # Loop para eliminar brilhos que nao estao em limbos
            flash_lados = [
                brilho
                for brilho in dict_cls_lados["flash"]
                if any(
                    auxiliar.in_caixa(limbo, brilho)
                    for limbo in dict_cls_lados["limbo"]
                )
            ]

            # ordena brilhos baseado na distancia para o limbo, considerando apenas aqueles que estao dentro de limbos
            flash_lados = auxiliar.sort_flash_bboxes(
                flash_lados, limbo_lados)[:1]
            try:
                flash_lados = flash_lados[0]
                limbo_lados = limbo_lados[0]
                auxiliar.caixas.get_dado(
                    num_paciente, [POSICAO, "FLASH".upper()]
                ).append(flash_lados)
                auxiliar.caixas.get_dado(
                    num_paciente, [POSICAO, "LIMBO".upper()]
                ).append(limbo_lados)
                return True
            except Exception as e:
                print(e)
                return False
        #
        else:
            self.no_detect += 1
            try:
                a = 1 / 0
            except Exception as e:
                print("NAO HOUVE DETECCAO NO PACIENTE")
            return False


# -----------------------------------------------------------------------------------------#
# OK
class Estrabismo:
    def __init__(self):
        self.olho_fix = ""

    def calculo_dioptrias(
        self,
        p_medio,
        centro_limbo,
        diam_limbo_desv: np.array,
        diam_limbo_fix: np.array,
        printar,
    ):
        """
        centro_limbo: list[Ponto]
        p_medio: list[Ponto]
        """
        desv_horiz = abs(p_medio[0] - centro_limbo[0])
        desv_vert = abs(p_medio[1] - centro_limbo[1])

        # fator de compensacao para ajustar dimensoes do limbo
        # fator_comp = diam_limbo_fix / diam_limbo_desv
        fator_comp = [1, 1]

        # Constante de conversao de pixel pra milimetro (baseado no diametro medio do limbo adulto)
        diam_adulto = 11
        pixelMM = diam_adulto / diam_limbo_fix

        # dp = constante de conversao de milimetros para dioptrias
        DiopMM = 15
        dh_diop = int(desv_horiz * pixelMM[0] * DiopMM * fator_comp[0])
        dv_diop = int(desv_vert * pixelMM[1] * DiopMM * fator_comp[1])

        if printar == True:
            print(f"Diop_H: {dh_diop}, Diop_V {dv_diop}", end=",  ")

        return dh_diop, dv_diop

    def classificacao_estrabismo(self, medio, limbo, printar):
        """Esotropia (ET)  Xcanto < Xlimbo
        Exotropia (XT)  Xcanto > Xlimbo
        Hipotropia (HoT)  Ycanto < Ylimbo
        Hipertropia (HT) Ycanto > Ylimbo"""

        estr_h = ""
        estr_v = ""

        # classificacao horizontal

        if medio[0] < limbo[0] * 0.9:
            estr_h = "ET"
        elif medio[0] > limbo[0] * 0.9:
            estr_h = "XT"
        else:
            estr_h = "ORTO"

        # classificacao vertical
        if medio[1] < limbo[1] * 0.9:
            estr_v = "HoT"
        elif medio[1] > limbo[1] * 0.9:
            estr_v = "HT"
        else:
            estr_v = "ORTO"

        if printar == True:
            print(f"DH: {estr_h} DV: {estr_v}", end="  ")
        return

    def estrabismo_flash(self, brilho, centro_limbos, image, diam, printar):
        # posicao 0 e 1 sendo esquerda e direita respectivamente
        # definir olho fixador
        dist_pontos = list()
        print(brilho, centro_limbos)
        # usa distancia euclidiana
        for i in range(len(brilho)):
            limbo = np.array([centro_limbos[i][0], centro_limbos[i][1]])
            flash = np.array([brilho[i][0], brilho[i][1]])
            # print(flash[0], flash[1])
            # primeiro = vermelho segundo = azul

            d = np.linalg.norm(limbo - flash)
            dist_pontos.append(d)
        index_min = dist_pontos.index(min(dist_pontos))
        olho_fix = ""
        # referencia deve ser esquerda e direita da imagem, por compatibilidade
        if index_min == 1:
            olho_desvio = "OD"
        else:
            olho_desvio = "OE"
        # olho a ser analisado
        ponto_flash = brilho[1 - index_min]
        ponto_flash = [
            int(ponto_flash[0]),
            int(ponto_flash[1]),
            int(ponto_flash[2]),
            int(ponto_flash[3]),
        ]
        # print(f"MEDIOS: {medio.x}, {medio.y}")
        # desenha brilho
        x1, y1 = (
            ponto_flash[0] - ponto_flash[2] // 2,
            ponto_flash[1] - ponto_flash[3] // 2,
        )
        x2, y2 = (
            ponto_flash[0] + ponto_flash[2] // 2,
            ponto_flash[1] + ponto_flash[3] // 2,
        )
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      color=(0, 255, 0), thickness=2)

        limbo = centro_limbos[1 - index_min]
        limbo = [int(limbo[0]), int(limbo[1]), int(limbo[2]), int(limbo[3])]
        diam_limbo_desv = np.array(diam[1 - index_min])
        diam_limbo_fix = np.array(diam[index_min])
        # desenha limbo
        cv2.circle(
            image, (limbo[0], limbo[1]), radius=3, color=(255, 0, 0), thickness=2
        )
        x1, y1 = limbo[0] - limbo[2] // 2, limbo[1] - limbo[3] // 2
        x2, y2 = limbo[0] + limbo[2] // 2, limbo[1] + limbo[3] // 2
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      color=(255, 0, 0), thickness=2)
        pos_desv = 1 - index_min

        dh_diop, dv_diop = self.calculo_dioptrias(
            ponto_flash, limbo, diam_limbo_desv, diam_limbo_fix, printar
        )
        self.classificacao_estrabismo(ponto_flash, limbo, printar)
        return dh_diop, dv_diop, pos_desv


# -----------------------------------------------------------------------------------------#
