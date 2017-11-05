#-*- coding: utf-8 -*-
import new_func
import pickle


#ОТКРЫВАЕМ ОСНОВНОЕ ВИДЕО
main_video = "main_video/reklama_2016.mp4"
main_stip_file = "main_video/reklama_2016.txt"

#СТРОИМ МОДЕЛЬ НА ОСНОВЕ ОСНОВНОГО ВИДЕО
model, frequency = new_func.built_model(main_stip_file)

#СТОИМ БАЗУ ИЗ РЕКЛАМЫ, ОПИСЫВАЕМ РЕКЛАМУ В ТЕРМИНАХ ОСНОВНОГО ВИДЕО
folder = "base_descr"
base = new_func.built_theBase(folder, model, frequency)
print(base)

base_file = open("base_descriptor.txt", "w")
pickle.dump(base, base_file)

model_file = open("model.txt", "w")
pickle.dump(model, model_file)