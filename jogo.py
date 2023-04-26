import os
import os.path
import sys

import cv2
import numpy as np

# funcao para redimensionar uma imagem com fator de escala
def resizeImage(img):
    return cv2.resize(img, (0, 0), None, 0.475, 0.475)

# funcao para desenhar texto em uma imagem
def drawOnScreen(img, text, origem, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(text), origem, font, 0.7, color, 2, cv2.LINE_AA)

# constantes para jogadas de Pedra, Papel e Tesoura
TESOURA = "TESOURA"
PEDRA = "PEDRA"
PAPEL = "PAPEL"
JOGADANAOIDENTIFICADA = "Jogada não identificada"

# Carregamento das imagens de referência
TEMPLATEPAPEL = resizeImage(cv2.imread("papel.png", 0))
TEMPLATETESOURA = resizeImage(cv2.imread("tesoura.png", 0))
TEMPLATEPEDRA = resizeImage(cv2.imread("pedra.png", 0))

# criacao de imagens invertidas para reconhecimento do jogador do lado dierito
REVERTTEMPLATEPAPEL = cv2.flip(TEMPLATEPAPEL, -1)
REVERTTEMPLATETESOURA = cv2.flip(TEMPLATETESOURA, -1)
REVERTTEMPLATEPEDRA = cv2.flip(TEMPLATEPEDRA, -1)

# Constantes para identificacao de cada jogador
PLAYERLEFT = "Jogador da esquerda"
PLAYERRIGHT = "Jogador da direita"

# Array para armazenar o placar dos jogadores
placar = [0, 0] # [PLAYER LEFT, PLAYER RIGHT]
colorBlack = [0, 0, 0]

# Variáveis para armazenar a jogada anterior de cada jogador, o jogador vencedor da ultima e a pontuação da ultima
lastMovePlayLeft = ""
lastMovePlayRight = ""
lastPlayerWin = ""
lastScoreView = ""

# funcao para detectar a jogada do jogador esquerdo
def movePlayerLeft(imgGray, imgRgb):
    # Match de cada imagem de referência com a imagem atual
    matchPapel = cv2.matchTemplate(imgGray, TEMPLATEPAPEL, cv2.TM_SQDIFF_NORMED)
    matchTesoura = cv2.matchTemplate(imgGray, TEMPLATETESOURA, cv2.TM_SQDIFF_NORMED)
    matchPedra = cv2.matchTemplate(imgGray, TEMPLATEPEDRA, cv2.TM_SQDIFF_NORMED)

    # Encontra a posicao da jogada correspondente com menor erro de correspondência
    minMatchValuePapel, _, positionMatchPapel, _ = cv2.minMaxLoc(matchPapel)
    minMatchValueTesoura, _, positionMatchTesoura, _ = cv2.minMaxLoc(matchTesoura)
    minMatchValuePedra, _, positionMatchPedra, _ = cv2.minMaxLoc(matchPedra)

    # Obtém a altura de cada imagem de ref para desenhar o texto abaixo da jogada
    _, heigthTemplatePapel = TEMPLATEPAPEL.shape[::-1]
    _, heigthTemplateTesoura = TEMPLATETESOURA.shape[::-1]
    _, heigthTemplatePedra = TEMPLATEPEDRA.shape[::-1]
    
    # Verifica qual jogada corresponde ao template com o menor valor de matching
    # Se o valor de matching for menor que o limiar de matching definido, considera a jogada identificada
    # REGRAS: < 0.019 = PAPEL, < 0.030 = TESOURA, < 0.0098 = PEDRA
    # Desenha a jogada na tela e retorna a jogada identificada e a posicao onde foi identificada
    # Caso contrário, retorna uma jogada não identificada e a posição (0, 0)
    if minMatchValuePapel < 0.019:
        drawPosition = (positionMatchPapel[0], positionMatchPapel[1] + heigthTemplatePapel + 30)
        drawOnScreen(imgRgb, PAPEL, drawPosition, colorBlack)
        return [PAPEL, positionMatchPapel]
    elif minMatchValueTesoura < 0.030:
        drawPosition = (positionMatchTesoura[0], positionMatchTesoura[1] + heigthTemplateTesoura + 30)
        drawOnScreen(imgRgb, TESOURA, drawPosition, colorBlack)
        return [TESOURA, positionMatchTesoura]
    elif minMatchValuePedra < 0.0098: 
        drawPosition = (positionMatchPedra[0], positionMatchPedra[1] + heigthTemplatePedra + 30)
        drawOnScreen(imgRgb, PEDRA, drawPosition, colorBlack)
        return [PEDRA, positionMatchPedra]
    else:
        return [JOGADANAOIDENTIFICADA, [0, 0]]

# Define a funcao que obtem a jogada do jogador da direita
# a funcao eh quase igual a função movePlayerLeft, porem usa os templates invertidos para reconhecer as jogadas
def movePlayerRight(imgGray, imgRgb):
    matchPapel = cv2.matchTemplate(imgGray, REVERTTEMPLATEPAPEL, cv2.TM_SQDIFF_NORMED)
    matchTesoura = cv2.matchTemplate(imgGray, REVERTTEMPLATETESOURA, cv2.TM_SQDIFF_NORMED)
    matchPedra = cv2.matchTemplate(imgGray, REVERTTEMPLATEPEDRA, cv2.TM_SQDIFF_NORMED)

    minMatchValuePapel, _, positionMatchPapel, _ = cv2.minMaxLoc(matchPapel)
    minMatchValueTesoura, _, positionMatchTesoura, _ = cv2.minMaxLoc(matchTesoura)
    minMatchValuePedra, _, positionMatchPedra, _ = cv2.minMaxLoc(matchPedra)

    _, heigthTemplatePapel = REVERTTEMPLATEPAPEL.shape[::-1]
    _, heigthTemplateTesoura = REVERTTEMPLATETESOURA.shape[::-1]
    _, heigthTemplatePedra = REVERTTEMPLATEPEDRA.shape[::-1]
    
    if minMatchValuePapel < 0.019:
        drawPosition = (positionMatchPapel[0], positionMatchPapel[1] + heigthTemplatePapel + 30)
        drawOnScreen(imgRgb, PAPEL, drawPosition, colorBlack)
        return [PAPEL, positionMatchPapel]
    elif minMatchValueTesoura < 0.030:
        drawPosition = (positionMatchTesoura[0] , positionMatchTesoura[1] + heigthTemplateTesoura + 30)
        drawOnScreen(imgRgb, TESOURA, drawPosition, colorBlack)
        return [TESOURA, positionMatchTesoura]
    elif minMatchValuePedra < 0.0098: 
        drawPosition = (positionMatchPedra[0] , positionMatchPedra[1] + heigthTemplatePedra + 30)
        drawOnScreen(imgRgb, PEDRA, drawPosition, colorBlack)
        return [PEDRA, positionMatchPedra]
    
    return [JOGADANAOIDENTIFICADA, [0, 0]]

# Define a função que calcula o placar da partida
# Recebe como entrada as jogadas dos jogadores e verifica qual jogador venceu a rodada
# Atualiza o placar e retorna uma mensagem indicando o vencedor da rodada e o placar atual
def score(movePlayerLeft, movePlayerRight):
    if (movePlayerLeft == TESOURA and movePlayerRight == PAPEL) or \
        (movePlayerLeft == PAPEL and movePlayerRight == PEDRA) or \
        (movePlayerLeft == PEDRA and movePlayerRight == TESOURA):
        placar[0] += 1
        scoreView = str("Placar: ") + str(placar)
        return  [f"{PLAYERLEFT} VENCEU", scoreView]
    elif (movePlayerLeft == PAPEL and movePlayerRight == TESOURA) or \
        (movePlayerLeft == PEDRA and movePlayerRight == PAPEL) or \
        (movePlayerLeft == TESOURA and movePlayerRight == PEDRA):
        placar[1] += 1
        scoreView = str("Placar: ") + str(placar)
        return [f"{PLAYERRIGHT} VENCEU", scoreView]
    else:
        scoreView = str("Placar: ") + str(placar)
        return ["* JOGADORES EMPATARAM *", scoreView]

# Define a funcao que verifica se ocorreu uma nova rodada (ou seja, se as jogadas dos jogadores mudaram)
# Recebe como entrada as jogadas dos jogadores e verifica se são diferentes das jogadas da rodada anterior
# Retorna True se ocorreu uma nova rodada, False caso contrário
def newRound(movePlayLeft, movePlayRight):
    global lastMovePlayLeft
    global lastMovePlayRight

    if movePlayLeft != lastMovePlayLeft or movePlayRight != lastMovePlayRight:
        lastMovePlayLeft = movePlayLeft
        lastMovePlayRight = movePlayRight
        return True
    return False

# Define a funcao que formata o frame para exibição na tela
# Processa o frame para obter as jogadas dos jogadores, calcular o placar e desenhar as informações na tela
# Retorna o frame processado com as informações desenhadas na tela
def formatFrame(img):
    global lastPlayerWin
    global lastScoreView

    # Redimensiona a imagem para uma largura máxima de 640 pixels
    imgScaled = resizeImage(img)

    # Converte a imagem para escala de cinza
    imgGray = cv2.cvtColor(imgScaled, cv2.COLOR_BGR2GRAY)

    imgWidth = imgScaled.shape[1]

    # obtem as jogadas dos jogadores a partir do frame processado
    movePlayLeft, matchPositionLeft = movePlayerLeft(imgGray, imgScaled)
    movePlayRight, matchPositionRight = movePlayerRight(imgGray, imgScaled)

    # Verifica se ocorreu uma nova rodada
    isNewRound = newRound(movePlayLeft, movePlayRight)

    # Se ocorreu uma nova rodada, calcula o placar e atualiza as informações a serem exibidas na tela
    if isNewRound:
        playerWin, scoreView = score(movePlayLeft, movePlayRight)
        lastPlayerWin = playerWin
        lastScoreView = scoreView

    # Desenha as info a serem exibidas na tela (jogadas dos jogadores, placar e mensagem de vencedor da rodada)
    drawOnScreen(imgScaled, lastScoreView, (int(imgWidth / 2) - 120, 50), colorBlack)
    drawOnScreen(imgScaled, lastPlayerWin, (int(imgWidth / 2) - 190, 90), colorBlack)
    drawOnScreen(imgScaled, PLAYERLEFT, (matchPositionLeft[0], (matchPositionLeft[1] - 30)), colorBlack)
    drawOnScreen(imgScaled, PLAYERRIGHT, (matchPositionRight[0], (matchPositionRight[1] - 30)), colorBlack)

    # Retorna o frame processado com as informações desenhadas na tela
    return imgScaled

vc = cv2.VideoCapture("pedra-papel-tesoura.mp4")
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # Formata o frame capturado pela câmera ou lido do arquivo de vídeo para exibição na tela
    img = formatFrame(frame)

    # Exibe o frame processado na tela
    cv2.imshow("preview", img)

    # Lê o próximo frame do arquivo de vídeo ou da câmera
    rval, frame = vc.read()

    # Aguarda a tecla ESC para interromper a execução
    key = cv2.waitKey(20)
    if key == 27:
        break

# Encerra a janela de exibição e libera a câmera ou o arquivo de vídeo
cv2.destroyWindow("preview")
vc.release()

# Exibe o resultado final da partida
if(placar[0] > placar[1]):
    print(f"{PLAYERLEFT} GANHOU com {placar[0]} vitórias" )
else:
    print(f"{PLAYERRIGHT} GANHOU com {placar[1]} vitórias" )




