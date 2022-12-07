# Project3_0 General 
# Part №1 Creation Face_DataSet

import cv2  # импорт библиотеки openCV
import numpy as np  # импорт библиотеки numpy
import os   # импорт модуля OS для работы с операционной системой

cam = cv2.VideoCapture(0)  # создаём объект для захвата видео
face_detector = cv2.CascadeClassifier('C:\XML\haarcascade_frontalface_default.xml')  # создаём объект, в который загружаем каскад Хаара из папки

face_id = "Michael"  # Вводим id лица, которое добавляется в имя и потом будет использовать в распознавании
print("\n [INFO] Инициализация захвата лица. Смотрите в камеру и ждите …")
count = 0

while True:                # организация цикла по обработке видеокадров
    ret, img = cam.read()  # передача считанных кадров в переменную
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # преобразование цветного изображения чёрно-белое и передача в переменную
    faces = face_detector.detectMultiScale(gray, 1.3, 5)  # распознавание лица в кадре в градациях серого
    for (x, y, w, h) in faces:                            # рисование прямоугольной рамки вокруг лиц
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1
        
        cv2.imwrite('C:\\DataSet\\user.' + str(face_id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])  # сохранение области лица в указанном месте с идентификаторами
    cv2.imshow('image', img)              # показ обработанного кадра (заголовок, значение переменной)
    k = cv2.waitKey(100) & 0xff  # проверка не нажата пользователем клавиша 'ESC'
    if k == 27:
        break
    elif count >= 60:  # Если сохранили 60 изображений выход.
        break

print("\n Программа завершена")
cam.release()                    # деактивация камеры
cv2.destroyAllWindows()          # закрыть все окна когда обработка окончена

# Part №2 Face_Lerning

path = 'C:\\DataSet\\'  # задние имени папки, где хранится набор с тренировочными фото
recognizer = cv2.face.LBPHFaceRecognizer_create()  # cоздание объекта распознавания лиц на основе класса LBPHFaceRecognizer


def getImagesAndLabels(path):             # функция чтения изображений из папки из папки с тренировочными фото
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # cоздание списка файлов в папке patch
    face = []  # список для хранения массива фото
    ids = []   # список для хранения идентификаторов лиц
    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # переводим изображение, тренер принимает изображения в оттенках серого
        face.append(img)  # записываем тренировочное фото в массив
        id = int(os.path.split(imagePath)[-1].split(".")[2])  # получаем идентификатор фото из его названия
        ids.append(id)  # записываем идентификатор тренировочного фото в массив
    return face, ids

faces, ids = getImagesAndLabels(path)   # чтение тренировочного набора фотографий из папки path
recognizer.train(faces, np.array(ids))  # тренировка модели распознования
recognizer.write('face_Michael.yml')    # сохранение результата тренировки (модели)

# Part №3 Recognition_Face

recognizer = cv2.face.LBPHFaceRecognizer_create()  # создание объекта recognizer
recognizer.read('face_Michael.yml')                # загрузка в этот объект предварительно обученной модели Michael
cascadePath = 'C:\XML\haarcascade_frontalface_default.xml'  # определение пути к расположению примитивам Хаара
faceCascade = cv2.CascadeClassifier(cascadePath)            # расписывание лиц на основе примитивов Хаара

font = cv2.FONT_HERSHEY_COMPLEX  # выбор тип шрифта

names = ['None', 'Michael']  # создание списока имен для идентификации (не распознанные - None, распознанные - Michael)

cam = cv2.VideoCapture(0)  # активация камеры
cam.set(3, 640)            # установка ширины видеокадра
cam.set(4, 480)            # установка высоты видеокадра

while True:                # организация цикла по обработке захваченных видеокадров
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # преобразование цветного изображения в градации серого      
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10),)  # распознавание лиц в текущем кадре с заданными параметрами

    for (x, y, w, h) in faces:     # организация цикла для рисования прямоугольных границ вокруг лиц         
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # размеры и красный цвет рамки, толщина - 2
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # возврат функцией идентификатора и степени достоверности
       
        if (confidence < 90):   # поверка того, что лицо распознано
            id_obj = names[1]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id_obj = names[0]
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id_obj), (x + 5, y - 5), font, 1, (255, 255, 255), 2)        # метод вывода на экран информации идентификатора с указанными параметрами
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)  # используем метод для вывода на экран информации о степени достоверности 
                                                                                          # с указанными параметрами
    cv2.imshow('camera', img)  # показ изображения

    k = cv2.waitKey(10) & 0xff  # проверка нажатия клавиши 'ESC' для выхода
    if k == 27:
        break

cam.release()            # деактивация камеры
cv2.destroyAllWindows()  # закрытие всех окон
