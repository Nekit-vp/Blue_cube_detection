import cv2
import numpy
import math as mat


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(numpy.dot(d1, d2) / numpy.sqrt(numpy.dot(d1, d1) * numpy.dot(d2, d2)))


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50,
                                apertureSize=5)  # Находит ребра в изображении, используя алгоритм Canny с пользовательским градиентом изображения.
                bin = cv2.dilate(bin, None)  # Расширяет изображение с помощью определенного элемента структурирования.
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255,
                                             cv2.THRESH_BINARY)  # Функция преобразует изображение в оттенках серого в двоичное изображение
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                                    cv2.CHAIN_APPROX_SIMPLE)  # Функция извлекает контуры из бинарного изображения с помощью алгоритма
            for cnt in contours:  # проходим по всем контурам, которые получились
                cnt_len = cv2.arcLength(cnt,
                                        True)  # Вычисляет контурный Периметр или длину кривой,  параметр true отвечает за замкнутость

                # Функция cv:: approxPolyDP аппроксимирует кривую или полигон другой кривой / полигоном с меньшим количеством вершин так, чтобы расстояние между ними было меньше или равно указанной точности. Он использует алгоритм Дугласа-Пекера
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) > 3  and len(cnt) < 7 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    # смотрим на количество углов, минимум их должны быть 4 до 6
                    cnt = cnt.reshape(-1, 2)
                    max_cos = numpy.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.8:
                        squares.append(cnt)

    return squares


low_blue = numpy.array((90, 20, 20), numpy.uint8)
high_blue = numpy.array((140, 255, 255), numpy.uint8)

path = 'picture/3.jpg'
src = cv2.imread(path)

cv2.imshow('исходное', src)

img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
mask_blue = cv2.inRange(img_hsv, low_blue, high_blue)

result = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_blue)
result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

squares = find_squares(result)
cv2.imshow('with blue color', result)
cv2.drawContours(src, squares, -1, (0, 255, 0), 4)
cv2.imshow('image', src)
cv2.waitKey(0)
