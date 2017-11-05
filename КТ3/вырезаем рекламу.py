#-*- coding: utf-8 -*-

import pickle
import new_func


#ОТКРЫВАЕМ ОСНОВНОЕ ВИДЕО
main_video = "main_video/reklama_2016.mp4"
main_stip_file = "main_video/reklama_2016.txt"

base = pickle.load(open("base_descriptor.txt"))
model = pickle.load(open("model.txt"))

#ПОЛУЧАЕМ ОПИСАНИЕ ОСНОВНОГО ВИДЕО :ВРЕМЯ ПОЯВЛЕНИЯ И ТИП КАЖДОЙ ЕГО ТОЧКИ
points_coordinates, frequency = new_func.stip_analysis(main_stip_file, model)
main_times, main_labels = new_func.video_to_unique_sequence(points_coordinates, frequency)

whole_movie_len = new_func.movie_len(main_video)


#ИДЕМ ПО БАЗЕ И ПРОВЕРЯЕМ ЕСТЬ ЛИ ФРАГМЕНТЫ ИЗ БАЗЫ В ОСНОВНОМ ВИДЕО
founded_reklama =[]
for filename in base.keys():
    max_score, subseqs = new_func.equal_sequence(main_labels, base[filename])
    slash = filename.find("/")
    format = filename.find(".txt")
    reklama = "base_video" + filename[slash:format] + ".mp4"
    print(reklama)
    fragment_len = new_func.movie_len(reklama)

    for ((a_s, a_e), (b_s, b_e)) in subseqs:
        start_time = main_times[a_s] * whole_movie_len
        end_time = main_times[a_e] * whole_movie_len if (a_e < len(main_times)) else whole_movie_len

        contemporaneity_len = end_time - start_time

        print(fragment_len)
        print(contemporaneity_len)
        print(main_times[a_s] * whole_movie_len, main_times[a_e] * whole_movie_len)
        if (abs(contemporaneity_len - fragment_len) < 0.1 * fragment_len):
            print("YES")
            founded_reklama.append((start_time, end_time))

(founded_reklama).sort(lambda (x, y): x[0] < y[0])

print(founded_reklama)

new_func.cut_the_fragment(founded_reklama, main_video, whole_movie_len)




