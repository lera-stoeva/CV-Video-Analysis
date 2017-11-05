#-*- coding: utf-8 -*-
import new_func
import pickle
import datetime


#ОТКРЫВАЕМ ОСНОВНОЕ ВИДЕО
main_video = "main_video2/Great_gatsby.mp4"
main_stip_file = "main_video2/Great_gatsby.txt"

#СТРОИМ МОДЕЛЬ НА ОСНОВЕ ОСНОВНОГО ВИДЕО
#model, frequency = new_func.built_model(main_stip_file, 100)

#f = open("Form_the_base/movie_model.txt", "w")
#pickle.dump(model, f)
#f.close()

model = pickle.load(open("Form_the_base/movie_model.txt"))

#ПОЛУЧАЕМ ОПИСАНИЕ ОСНОВНОГО ВИДЕО :ВРЕМЯ ПОЯВЛЕНИЯ И ТИП КАЖДОЙ ЕГО ТОЧКИ
points_coordinates, frequency = new_func.stip_analysis(main_stip_file, model)
main_times, main_labels = new_func.video_to_unique_sequence(points_coordinates, frequency)

#ЗАПУСКАЕМ ПОИСК ПОВТОРОВ В ПОСЛЕДОВАТЕЛЬНОСТИ ИЗ ТИПОВ ТОЧЕК, ПОЛУЧАЕМ МАКСИМАЛЬНЫЙ ВЕС И КООРДИНАТЫ НАЧАЛА И КОНЦА ВСЕХ НАЙДЕНЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ С ЭТИМ ВЕСОМ
max_score, subseqs = new_func.search_repetitions(main_labels)

#НАХОДИМ ДЛИНУ НАШЕГО ВИДЕО
whole_movie_len = new_func.movie_len(main_video)

#ПЕРЕОБРАЗУЕМ КООРДИНАТЫ НАЧАЛА И КОНЦА КО ВРЕМЕНИ НАЧАЛА И КОНЦА НАЙДЕНЫХ ПОВТОРОВ
#ЕСЛИ ПОВТОР ДЛИТСЯ 30 СЕКУНД (+- 10), ТО СКОРЕЕ ВСЕГО ЭТО РЕКЛАМА, ЗНАЧИТ НАМ НАДО ЗАПОМНИТЬ ЭТОТ ВРЕМЕНОЙ ИНТЕРВАЛ. (РЕКЛАМА РЕДКО ДЛИТСЯ 5 СЕКУНД ИЛИ МИНУТУ)
good_intervals = []
for (a_s, a_e) in subseqs:
    start_time = main_times[a_s] * whole_movie_len
    end_time = main_times[a_e] * whole_movie_len if (a_e < len(main_times)) else whole_movie_len
    contemporaneity_len = end_time - start_time
    print(contemporaneity_len)

    if (abs(contemporaneity_len - 30) < 10):
        print("YES")
        start = datetime.timedelta(seconds = start_time)
        end = datetime.timedelta(seconds = end_time)
        normal_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min) + start #получаем дату со временем начала рекламы
        normal_end = datetime.datetime.combine(datetime.date.today(), datetime.time.min) + end
        normal_start.time()
        normal_end.time()

        print(a_s, normal_start.isoformat())
        print(a_e, normal_end.isoformat())

        good_intervals.append((a_s, a_e))


#РАЗБИВЕМ ВСЕ ИНТЕРВАЛЫ НА ГРУППЫ. В ОДНУ ГРУППУ ПОПАДУТ ПЕРЕСЕКАЮЩИЕСЯ МЕЖДУ СОБОЙ ИНТЕРВАЛЫ
interval_groups = {}
for i in range(0, len(good_intervals) - 1):
    interval_groups[i] = {i}
    for j in range(i + 1, len(good_intervals)):
        if (new_func.intersection(good_intervals[i], good_intervals[j])):
            flag = 0
            for k in range(0, i):
                if (i in interval_groups[k]):
                    interval_groups[k].add(j)
                    flag = 1
                    break
            if (flag == 0):
                interval_groups[i].add(j)
interval_groups[len(good_intervals) - 1] = {len(good_intervals) - 1}
print(interval_groups)

for i in range(0, len(good_intervals)):
    for j in interval_groups[i]:
        if (j != i and len(interval_groups[j]) == 1):
            interval_groups[j].clear()

#В РУЗУЛЬТАТЫ ЗАПИСЫВАЕМ ПЕРЕСЕЧЕНИЕ ВСЕХ ИНТЕРВАЛОВ ИЗ ОДНОЙ ГРУППЫ
results = []
for i in interval_groups.keys():
    if (len(interval_groups[i]) > 0):
        interval = good_intervals[list (interval_groups[i])[0]]
        for j in interval_groups[i]:
            interval = new_func.merge(interval, good_intervals[j])
        results.append(interval)

#ВИЗУАЛИЗИРУЕМ: ИЗ ОСНОВНОГО ВИДЕО ВЫРЕЗАЕМ ВСЕ ИНТЕРВАЛЫ ПО ВРЕМЕНИ, КОТОРЫЕ МЫ ЗАПИСАЛИ В РЕЗУЛЬТАТЫ. ВЫРЕЗАНУЮ РЕКЛАМУ ОТПРАВЛЯЕМ В ПАПКУ "ФОРМИРУЕМ БАЗУ"
new_func.extract_reklama(results, main_times, main_video, whole_movie_len)
print(interval_groups)
print(results)


