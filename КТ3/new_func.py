#-*- coding: utf-8 -*-
from sklearn import cluster
import numpy as np
import collections
import os
import cv2
import ffmpy
import subprocess
import datetime
import math


def stip_analysis(filename, model):
    with open(filename) as f:
        lines = f.read().split('\n')
    f.close()
    features = []
    coordinates = []
    for i in range(3, len(lines) - 1):
        values = (list(map(np.float32, lines[i].split())))
        features.append(values[7:169])
        coordinates.append(values[1:4])# достаем х у т
    features = np.array(features)

    labels = model.predict(features).astype(int) # взяли новое(то же) видео и распределили его точки по нашим 10 кластерам
    frequency = {x: np.count_nonzero(labels == x) * 1.0 / labels.shape[0] for x in labels}

    for i in range(0, len(coordinates)):
        coordinates[i].append(labels[i])
    coordinates.sort(key=lambda x: x[2]) # сортируем по времени
    coordinates = np.array(coordinates) # обратно в массив
    return coordinates, frequency # массив масивов (х у т тип точки)  и частота появления каждой точки в нормированом виде


def deletion_byRow (distance, row, column):
    deletion = distance[row - 1][column] - 1  # функция штрафа пропуска 1 элемента
    traceback_start = (row - 1, column)  # переход при удалении из первой последовательности (треугольник вниз)
    for k in range(2, row + 1):
        current_deletion = distance[row - k][column] - k
        if current_deletion > deletion:
            deletion = current_deletion
            traceback_start = (row - k, column)
    return deletion, traceback_start


def deletion_byColumn (distance, row, column):
    deletion = distance[row][column - 1] - 1
    traceback_start = (row, column - 1)
    for k in range(2, column + 1):
        current_deletion = distance[row][column - k] - k
        if current_deletion > deletion:
            deletion = current_deletion
            traceback_start = (row, column - k)
    return deletion, traceback_start

def subsequencies(distance, a_size, traceback, max_score, b_size = None):
    subseqs = []
    for row in range(1, a_size + 1):

        if (b_size != None):
            columns = b_size + 1
        else :
            columns = row

        for column in range(1, columns):
            if distance[row][column] != max_score:
                continue
            trace_i, trace_j = traceback[row][column]
            a_begin, b_begin = row - 1, column - 1
            while distance[trace_i][trace_j] != 0:
                a_begin, b_begin = trace_i - 1, trace_j - 1
                trace_i, trace_j = traceback[trace_i][trace_j]
            a_subseq = (a_begin, row)
            b_subseq = (b_begin, column)
            subseqs.append((a_subseq,
                            b_subseq))
    return subseqs

def equal_sequence(a, b, match=1, shift=-1):
    """Extracts similar subsequences"""
    a_size, b_size = len(a), len(b)
    distance = np.zeros((a_size + 1, b_size + 1), dtype=float)
    traceback = [[(0, 0)] * (b_size + 1)] # список из b_size + 1 пар 0-0 (список переходов)
    max_score = 0

    for row in range(1, a_size + 1):
        traceback_line = [(0, 0)] # почему начало матрицы, а не последовательности?
        for column in range(1, b_size + 1):
            trace_point = {}
            fee = match if a[row-1] == b[column-1] else shift
            trace_point[distance[row - 1][column - 1] + fee] = (row - 1, column - 1)
            row_deletion, row_traceback = deletion_byRow(distance, row, column)
            column_deletion, column_traceback = deletion_byColumn(distance, row, column)

            trace_point[row_deletion] = row_traceback
            trace_point[column_deletion] = column_traceback
            trace_point[0] = (0, 0)
            distance[row][column] = max(trace_point)
            traceback_line.append(trace_point[distance[row][column]])

            if distance[row][column] > max_score:
                max_score = distance[row][column]  # мы же ищем подпоследобательность максимальной длины

        traceback.append(traceback_line) # добавляем в общуюю

    subseqs = subsequencies(distance, a_size, traceback, max_score, b_size)
    return max_score, subseqs


def search_repetitions(a, match=1, shift=-1):
    a_size = len(a)
    distance = np.zeros((a_size + 1, a_size + 1), dtype=float)
    traceback = [[(0, 0)] * (a_size + 1)]   # список из b_size + 1 пар 0-0 (список переходов)

    for row in range(a_size + 1):
        for column in range(row, a_size + 1):
            distance[row][column] = -10 * a_size # заполняе верхнюю часть чем-то плохим

    # Filling the distance table and traceback
    for row in range(1, a_size + 1):
        traceback_line = [(0, 0)]  # почему начало матрицы, а не последовательности?
        for column in range(1, row):
            trace_point = {}
            fee = match if a[row - 1] == a[column - 1] else shift
            trace_point[distance[row - 1][column - 1] + fee] = (row - 1, column - 1)
            row_deletion, row_traceback = deletion_byRow (distance, row, column)
            # максимальное удаление (треугольник вниз)  для А и его стоимость

            column_deletion, column_traceback = deletion_byColumn (distance, row, column)
            # максимальное удаление (треугольник ввкрх)  для В и его стоимость

            trace_point[row_deletion] = row_traceback
            trace_point[column_deletion] = column_traceback
            trace_point[0] = (0, 0)
            distance[row][column] = max(trace_point)
            traceback_line.append(trace_point[distance[row][column]])
        traceback.append(traceback_line)  # добавляем в общуюю

    max_score = 0;
    start = math.ceil(len(a) / 3)
    k = 0;
    for i in range (int(start), a_size + 1):
        k += 1;
        for j in range (int(start), k):
            if (distance[i][j] > max_score):
                max_score = distance[i][j]

    # Results based on traceback
    subseqs = subsequencies(distance, a_size, traceback, max_score) # находим все подпоследовательности максимальной длины (то есть одной и  тойже) для а и б они должны быть равны
    #print(subseqs)
    result = set()
    for subseq in subseqs:
        result.add(subseq[0]) # берем только одну, таккака  аи б равны
    return max_score, list(result)


def video_to_unique_sequence(coordinates, frequency): # в каждом кадре ищем самую уникальную точку, тип которой находится в меньшенстве.
    time_cluster = {}
    timeNormal, clusters = [], []
    row = 0

    while row < coordinates.shape[0]:
        currentTime = coordinates[row][2]
        time_cluster[currentTime] = -1
        minFrequency = 1
        while row < coordinates.shape[0] and coordinates[row][2] == currentTime:
            cluster = coordinates[row][3]
            if minFrequency > frequency[int(cluster)]:
                minFrequency = frequency[int(cluster)]
                time_cluster[currentTime] = cluster
            row += 1
    time_cluster = collections.OrderedDict(sorted(time_cluster.items()))

    #print(time_cluster)
    return list(time_cluster.keys()), list(time_cluster.values()) # получаем два массива одной длины со временем и точкой


def built_model(main_stip_file, n_clusters = 10):
    file = open(main_stip_file, 'r')
    lines = file.readlines()
    matrix = [line.split() for line in lines[2:-1]]
    matrix = np.array(matrix)
    file.close()
    matrix = matrix[:, 9:171]
    matrix.astype(float)
    model = cluster.KMeans(n_clusters).fit(matrix)
    points_coordinates, frequency = stip_analysis(main_stip_file, model)
    return model, frequency

def built_theBase(folder, model, frequency):
    base = {}
    for root, dirs, files in os.walk(folder):  # os.walk(folder) возвращает путь, и файлы в папке
        for file in files:
            if (file[0] != "."):
                filename = os.path.join(root, file)  # получаем полный путь к файлу
                print(file)
                # получаем матрицу дискрипторов
                coordinates, _ = stip_analysis(filename, model)
                b_times, labels = video_to_unique_sequence(coordinates, frequency)  # частота остается от большого видео, так работает лучше
                base[filename] = labels
    return base


def movie_len(video):
    mov_t, error = ffmpy.FFmpeg(executable="ffprobe",
                                global_options="-v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1",
                                inputs={video: None}, outputs=None).run(stdout=subprocess.PIPE)
    whole_movie_len = float(mov_t)
    return(whole_movie_len)


def cut_the_fragment(founded_reklama, main_video, whole_movie_len):
    part_start = datetime.time.min
    counter = 0
    video_name = main_video.split('/')
    video_name = video_name[len(video_name) - 1]
    part = "fragments_without_ad/" + video_name + "_part"


    for (reklama_start, reklama_end) in founded_reklama:
        start = datetime.timedelta(seconds = reklama_start)
        end = datetime.timedelta(seconds = reklama_end)
        normal_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min) + start  # получаем дату со временем начала рекламы
        normal_end = datetime.datetime.combine(datetime.date.today(), datetime.time.min) + end
        part_end = normal_start.time()  # оставляем только время начала рекламы
        if (part_start < part_end):
            ffmpy.FFmpeg(global_options="-ss " + str(part_start) + " -to " + str(part_end) + " -c copy -y " + part + str(
                    counter) + ".mp4", inputs={main_video: None}).run()
            counter += 1
            part_start = normal_end.time()

    part_end = datetime.datetime.combine(datetime.date.today(), datetime.time.min) + datetime.timedelta(seconds = whole_movie_len)
    part_end = part_end.time()
    ffmpy.FFmpeg(global_options="-ss " + str(part_start) + " -to " + str(part_end) + " -c copy -y " + part + str(
        counter) + ".mp4", inputs={main_video: None}).run()
    counter

    file_with_bits = open("bits.txt", "w")
    for i in range(0, counter + 1):
        file_with_bits.write("file    '" + part + str(i) + ".mp4'\n")
    file_with_bits.close()

    ffmpy.FFmpeg(global_options="-f concat -safe 0 -i bits.txt -c copy -y ready_to_watch.mp4").run()


def merge((x1, x2), (y1, y2)):
    start = max(x1, y1)
    end = max(x2, y2)
    return (start, end)

def intersection((a_s, a_e), (b_s, b_e)):
    if (a_s <= b_s and b_s <= a_e):
        return True
    if (b_s <= a_s and a_s <= b_e):
        return True
    if (a_s <= b_e and b_e <= a_e):
        return True
    if (b_s <= a_e and a_e <= b_e):
        return True

def extract_reklama(intervals, main_times, main_video, whole_movie_len):
    counter = 0
    video_name = main_video.split('/')
    video_name = video_name[len(video_name) - 1]
    part = "Form_the_base/" + video_name + "reklama"

    for (reklama_start, reklama_end) in intervals:
        start = datetime.timedelta(seconds = main_times[reklama_start] * whole_movie_len)
        end = datetime.timedelta(seconds = main_times[reklama_end] * whole_movie_len)
        normal_start = datetime.datetime.combine(datetime.date.today(),
                                                 datetime.time.min) + start  # получаем дату со временем начала рекламы
        normal_end = datetime.datetime.combine(datetime.date.today(), datetime.time.min) + end
        print(str(normal_start.time()), str(normal_end.time()))
        ffmpy.FFmpeg(global_options="-ss " + str(normal_start.time()) + " -to " + str(normal_end.time()) + " -c copy -y " + part + str(
                    counter) + ".mp4", inputs={main_video: None}).run()
        counter += 1