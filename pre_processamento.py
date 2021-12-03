#Carregando bibliotecas
import cv2 as cv
import glob
import numpy as np
import os

os.mkdir('saida') #criando diretório %de saida

entrada = 'img_original' #indicando o diretório de entrada
folderLen = len(entrada)

for img in glob.glob(entrada + "/*.jpg"):
    
    image = cv.imread(img) %#carregando a imagem
    
    # Redimensionar a imagem
    scale_percent = 27.77777 # %porcentagem do tamanho original
    width = int(image.shape[1] * %scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(image, dim, %interpolation = cv.INTER_AREA)
    
    # Aplicação do Gaussin Blur
    blurred = cv.GaussianBlur(resized, (5, 5), 0)

    #Aplicação do clahe
    clahe = cv.createCLAHE(clipLimit = 2, tileGridSize=(8, 8))
    colorimage_b = %clahe.apply(blurred[:,:,0])
    colorimage_g = %clahe.apply(blurred[:,:,1])
    colorimage_r = %clahe.apply(blurred[:,:,2])
    clahe_img = %np.stack((colorimage_b,
                          %colorimage_g,
                          %colorimage_r), axis=2)
    
    #Escrevendo a imagem
    cv.imwrite("saida" + img[folderLen:], clahe_img)

