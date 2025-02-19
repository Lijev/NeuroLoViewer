import json
import numpy as np
import tensorflow as tf
import os
import shutil

# 1. Загрузка данных из IDS.json
def load_data(filepath="IDS.json"):
    """Загружает данные из JSON-файла."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            X = np.array(data['X']).T  # Транспонируем X
            Y = np.array(data['Y']).T  # Транспонируем Y

            # Нормализация данных (пример)
            X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True) # Стандартизация
            Y = (Y - np.mean(Y, axis=1, keepdims=True)) / np.std(Y, axis=1, keepdims=True)  # Стандартизация


            return X, Y, data  # Возвращаем и data
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filepath}' не найден.")
        return None, None, None
    except json.JSONDecodeError:
        print(f"Ошибка: Невозможно декодировать JSON из файла '{filepath}'. Проверьте формат файла.")
        return None, None, None
    except KeyError as e:
        print(f"Ошибка: Ключ '{e}' отсутствует в JSON-файле.  Убедитесь, что JSON содержит массивы 'X' и 'Y'.")
        return None, None, None

# Функция для сохранения данных в IDS.json
def save_data(filepath="IDS.json", data=None):
    """Сохраняет обновленные данные в JSON-файл."""
    if data is None:
        print("Ошибка: Нет данных для сохранения.")
        return

    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)  # Записываем с отступами для читаемости
        print(f"Данные сохранены в {filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении данных: {e}")

# Функция для очистки экрана
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Функция для вывода справки
def print_help():
    print("Список команд:")
    print("help - Объяснение указанной команды (по умолчанию выводит этот список)")
    print("save - Сохранение модели с указанным именем и директорией (по умолчанию LoView в текущей директории)")
    print("load - Загрузка модели с указанным именем и директорией (по умолчанию LoView в текущей директории)")
    print("train - Тренировка текущей модели на указанное кол-во эпох (по умолчанию 100 эпох)")
    print("use - Использование нейросети (введите 3 числа, или 'all' для третьего, модель выдаст 4 числа)")
    print("show - Показывает 'правильные' ответы из датасета для заданных входных значений (аналогично 'use')")
    print("add - Добавление новых данных в датасет (только если X не существует)")
    print("edit - Редактирование существующих данных в датасете")
    print("remove - Удаление данных из датасета")
    print("delete - Удаление сохраненной модели")
    print("quit - Выход")

# Функция для сохранения модели
def save_model(model, name="LoView", directory="."):
    filepath = os.path.join(directory, name)
    model.save(filepath)
    print(f"Модель сохранена в {filepath}")

# Функция для загрузки модели
def load_model(name="LoView", directory="."):
    filepath = os.path.join(directory, name)
    try:
        model = tf.keras.models.load_model(filepath)
        print(f"Модель загружена из {filepath}")
        return model
    except OSError:
        print(f"Ошибка: Не удалось загрузить модель из {filepath}.  Проверьте имя и путь.")
        return None

# Функция для обучения модели
def train_model(model, X, Y, epochs=100):
    print("Начинаем обучение...")
    model.fit(X.T, Y.T, epochs=epochs, batch_size=32, verbose=1)
    print("Обучение завершено.")

# Функция для использования модели
def use_model(model):
    try:
        num1 = float(input("Введите первое число: "))
        num2 = float(input("Введите второе число: "))
        num3 = input("Введите третье число (или 'all'): ").lower()

        if num3 == "all":
            third_numbers = np.linspace(1, 10, 10)  # Генерируем 10 чисел от 1 до 10
            for i, third_number in enumerate(third_numbers):
                new_example = np.array([[num1], [num2], [third_number]])
                new_example = new_example.T
                prediction = model.predict(new_example)
                print(f"Результат для третьего числа {third_number:.2f}: {prediction}") # Выводим с форматированием для красоты
        else:
            try:
                num3 = float(num3)
                new_example = np.array([[num1], [num2], [num3]])
                new_example = new_example.T
                prediction = model.predict(new_example)
                print(f"Результат: {prediction}")
            except ValueError:
                print("Ошибка: Пожалуйста, введите корректное число для третьего числа.")

    except ValueError:
        print("Ошибка: Пожалуйста, введите корректные числа для первого и второго чисел.")

# Функция для показа "правильных" ответов из датасета
def show_data(X, Y):  # Добавим X и Y как аргументы
    try:
        num1 = float(input("Введите первое число: "))
        num2 = float(input("Введите второе число: "))
        num3 = input("Введите третье число (или 'all'): ").lower()

        if num3 == "all":
            third_numbers = np.linspace(1, 10, 10)  # Генерируем 10 чисел от 1 до 10
            for i, third_number in enumerate(third_numbers):
                input_values = np.array([num1, num2, third_number]) # Создаём вектор входных значений
                found = False
                for j in range(X.shape[1]):
                    if np.allclose(X[:, j], input_values): #Сравниваем с каждым столбцом X
                        print(f"Правильный ответ для третьего числа {third_number:.2f}: {Y[:, j]}")
                        found = True
                        break  # Нашли соответствие, выходим из внутреннего цикла

                if not found:
                    print(f"Данные для третьего числа {third_number:.2f} не найдены в датасете.")


        else:
            try:
                num3 = float(num3)
                input_values = np.array([num1, num2, num3])
                found = False
                for j in range(X.shape[1]):
                     if np.allclose(X[:, j], input_values):
                         print(f"Правильный ответ: {Y[:, j]}")
                         found = True
                         break
                if not found:
                    print("Данные не найдены в датасете.")



            except ValueError:
                print("Ошибка: Пожалуйста, введите корректное число для третьего числа.")

    except ValueError:
        print("Ошибка: Пожалуйста, введите корректные числа для первого и второго чисел.")

def add_data(X, Y, data):
    """Добавляет новые данные в датасет, если X не существует."""
    try:
        num1 = float(input("Введите первое число для X: "))
        num2 = float(input("Введите второе число для X: "))
        num3_input = input("Введите третье число для X (или 'all'): ").lower()

        if num3_input == "all":
            num3_values = np.linspace(1, 10, 10)  # Создаем 10 значений для третьего числа

            # Проверка на существование
            for num3 in num3_values:
                input_values = [num1, num2, num3]
                for existing_x in data['X']:
                    if np.allclose(existing_x, input_values):
                        print(f"Ошибка: Данные X: {input_values} уже существуют.  Нельзя добавить дубликат.")
                        return  # Выходим из функции, если нашли дубликат

            Y_values = []
            for i in range(len(num3_values)): # запрашиваем для каждого X свой Y
                 y1 = float(input(f"Введите первое число для Y (для X: {num1}, {num2}, {num3_values[i]}): "))
                 y2 = float(input(f"Введите второе число для Y (для X: {num1}, {num2}, {num3_values[i]}): "))
                 y3 = float(input(f"Введите третье число для Y (для X: {num1}, {num2}, {num3_values[i]}): "))
                 y4 = float(input(f"Введите четвертое число для Y (для X: {num1}, {num2}, {num3_values[i]}): "))
                 Y_values.append([y1, y2, y3, y4])

            confirmation = input("Вы уверены, что хотите добавить эти данные? (y/n): ").lower()

            if confirmation == "y":

                for i in range(len(num3_values)):
                    new_x = [num1, num2, num3_values[i]]
                    new_y = Y_values[i]

                    data['X'].append(new_x)
                    data['Y'].append(new_y)

                save_data(data=data)  # Сохраняем обновленные данные
                print("Данные успешно добавлены.")

            else:
                print("Добавление данных отменено.")


        else:

            num3 = float(num3_input)  # Преобразуем в число, если не "all"
            input_values = [num1, num2, num3]  # Формируем искомый X

            # Проверка на существование
            for existing_x in data['X']:
                if np.allclose(existing_x, input_values):
                    print(f"Ошибка: Данные X: {input_values} уже существуют. Нельзя добавить дубликат.")
                    return  # Выходим из функции

            y1 = float(input("Введите первое число для Y: "))
            y2 = float(input("Введите второе число для Y: "))
            y3 = float(input("Введите третье число для Y: "))
            y4 = float(input("Введите четвертое число для Y: "))


            confirmation = input("Вы уверены, что хотите добавить эти данные? (y/n): ").lower()

            if confirmation == "y":

                new_x = [num1, num2, num3]
                new_y = [y1, y2, y3, y4]

                data['X'].append(new_x)
                data['Y'].append(new_y)

                save_data(data=data)  # Сохраняем обновленные данные
                print("Данные успешно добавлены.")

            else:
                print("Добавление данных отменено.")

    except ValueError:
        print("Ошибка: Пожалуйста, введите корректные числа.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")



def edit_data(X, Y, data): # Добавьте data как аргумент
    """Редактирует существующие данные в датасете."""
    try:
        num1 = float(input("Введите первое число для поиска X: "))
        num2 = float(input("Введите второе число для поиска X: "))
        num3_input = input("Введите третье число для поиска X (или 'all'): ").lower()

        if num3_input == "all":
            third_numbers = np.linspace(1, 10, 10)
            input_values_list = [[num1, num2, num3] for num3 in third_numbers]  # Список для поиска
            # input_values = np.array([num1, num2, num3])  # Создаем массив numpy для поиска

            for input_values in input_values_list:
              found_index = None # Добавим индекс для отслеживания
              for i in range(len(data['X'])):
                if np.allclose(data['X'][i], input_values):
                  found_index = i
                  break

              if found_index is not None:
                  print(f"Найдено соответствие для X: {data['X'][found_index]}, Y: {data['Y'][found_index]}")

                  confirm_edit = input("Вы хотите отредактировать эти данные? (y/n): ").lower()
                  if confirm_edit == "y":

                      y1 = float(input("Введите новое первое число для Y: "))
                      y2 = float(input("Введите новое второе число для Y: "))
                      y3 = float(input("Введите новое третье число для Y: "))
                      y4 = float(input("Введите новое четвертое число для Y: "))
                      new_y = [y1, y2, y3, y4]

                      confirm_final = input("Вы уверены, что хотите заменить Y данными {} ? (y/n):".format(new_y)).lower() # добавили подтверждение

                      if confirm_final == "y":
                         data['Y'][found_index] = new_y
                         save_data(data=data)
                         print("Данные успешно отредактированы.")
                      else:
                        print("Редактирование отменено")


                  else:
                      print("Редактирование отменено.")
              else:
                   print("Данные не найдены в датасете.")


        else:
            try:
                num3 = float(num3_input)
                input_values = [num1, num2, num3] #  Преобразовываем к списку

                found_index = None  # Найденный индекс

                for i in range(len(data['X'])):  # Перебираем индексы
                  if np.allclose(data['X'][i], input_values):
                    found_index = i # запоминаем индекс
                    break


                if found_index is not None: # Если нашли соответствие
                  print(f"Найдено соответствие для X: {data['X'][found_index]}, Y: {data['Y'][found_index]}") #выводим

                  confirm_edit = input("Вы хотите отредактировать эти данные? (y/n): ").lower()
                  if confirm_edit == "y":

                      y1 = float(input("Введите новое первое число для Y: "))
                      y2 = float(input("Введите новое второе число для Y: "))
                      y3 = float(input("Введите новое третье число для Y: "))
                      y4 = float(input("Введите новое четвертое число для Y: "))
                      new_y = [y1, y2, y3, y4]

                      confirm_final = input("Вы уверены, что хотите заменить Y данными {} ? (y/n):".format(new_y)).lower() # добавили подтверждение

                      if confirm_final == "y":
                         data['Y'][found_index] = new_y
                         save_data(data=data)
                         print("Данные успешно отредактированы.")
                      else:
                        print("Редактирование отменено")


                  else:
                      print("Редактирование отменено.")

                else:
                    print("Данные не найдены в датасете.")


            except ValueError:
                print("Ошибка: Пожалуйста, введите корректное число для третьего числа.")

    except ValueError:
        print("Ошибка: Пожалуйста, введите корректные числа.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def remove_data(X, Y, data):
    """Удаляет данные из датасета."""
    try:
        num1 = float(input("Введите первое число для поиска X: "))
        num2 = float(input("Введите второе число для поиска X: "))
        num3_input = input("Введите третье число для поиска X (или 'all'): ").lower()

        if num3_input == "all":
            third_numbers = np.linspace(1, 10, 10)
            input_values_list = [[num1, num2, num3] for num3 in third_numbers]

            indices_to_remove = []
            for input_values in input_values_list:
                for i in range(len(data['X'])):
                    if np.allclose(data['X'][i], input_values):
                        indices_to_remove.append(i)

            if indices_to_remove:
                print("Найдены следующие соответствия:")
                for index in indices_to_remove:
                    print(f"X: {data['X'][index]}, Y: {data['Y'][index]}")

                confirmation = input("Вы уверены, что хотите удалить эти данные? (y/n): ").lower()
                if confirmation == "y":
                    # Удаляем элементы с конца, чтобы не нарушить индексы при удалении
                    for index in sorted(indices_to_remove, reverse=True):
                        del data['X'][index]
                        del data['Y'][index]
                    save_data(data=data)
                    print("Данные успешно удалены.")
                else:
                    print("Удаление данных отменено.")
            else:
                print("Данные не найдены в датасете.")

        else:
            try:
                num3 = float(num3_input)
                input_values = [num1, num2, num3]

                found_index = None
                for i in range(len(data['X'])):
                    if np.allclose(data['X'][i], input_values):
                        found_index = i
                        break

                if found_index is not None:
                    print(f"Найдено соответствие: X: {data['X'][found_index]}, Y: {data['Y'][found_index]}")
                    confirmation = input("Вы уверены, что хотите удалить эти данные? (y/n): ").lower()
                    if confirmation == "y":
                        del data['X'][found_index]
                        del data['Y'][found_index]
                        save_data(data=data)
                        print("Данные успешно удалены.")
                    else:
                        print("Удаление данных отменено.")
                else:
                    print("Данные не найдены в датасете.")

            except ValueError:
                print("Ошибка: Пожалуйста, введите корректное число для третьего числа.")

    except ValueError:
        print("Ошибка: Пожалуйста, введите корректные числа.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def delete_model(directory="."):
    """Удаляет сохраненную модель."""
    name = input("Введите имя модели для удаления (по умолчанию LoView): ") or "LoView"
    filepath = os.path.join(directory, name)
    try:
        shutil.rmtree(filepath)  # Используем rmtree для удаления директорий с содержимым
        print(f"Модель {name} успешно удалена из {directory}")
    except FileNotFoundError:
        print(f"Ошибка: Модель {name} не найдена в {directory}")
    except OSError as e:
        print(f"Ошибка при удалении модели: {e}")



# 2.  Основной код
if __name__ == '__main__':
    # Задаем размеры слоев
    n_input = 3  # 3 входные клетки
    n_hidden = 30000
    n_output = 4  # 4 выходные клетки

    # Загрузка данных
    X, Y, data = load_data()

    if X is None or Y is None or data is None:
        exit() # Завершаем программу, если не удалось загрузить данные

    # Проверка размерности данных
    #if X.shape[0] != n_input:
    #    raise ValueError(f"Размерность входных данных (X.shape[0]={X.shape[0]}) не соответствует ожидаемой (n_input={n_input]).")
    #if Y.shape[0] != n_output:
    #    raise ValueError(f"Размерность выходных данных (Y.shape[0]={Y.shape[0]}) не соответствует ожидаемой (n_output={n_output}).")

    # 3. Создание модели TensorFlow
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n_hidden, activation='relu', input_shape=(n_input,)),
        tf.keras.layers.Dense(n_output)  # Линейная активация для регрессии
    ])

    # 4. Компиляция модели
    model.compile(optimizer='adam', loss='mse')

    # Приветствие
    print("Добро пожаловать!")
    print_help()

    # Цикл обработки команд
    while True:
        command = input("Введите команду (help для списка команд): ").lower()
        clear_screen()

        if command == "help":
            print_help()
        elif command == "save":
            name = input("Введите имя модели (по умолчанию LoView): ") or "LoView"
            directory = input("Введите директорию сохранения (по умолчанию текущая): ") or "."
            save_model(model, name, directory)
        elif command == "load":
            name = input("Введите имя модели (по умолчанию LoView): ") or "LoView"
            directory = input("Введите директорию загрузки (по умолчанию текущая): ") or "."
            loaded_model = load_model(name, directory)
            if loaded_model:
                model = loaded_model # Заменяем текущую модель загруженной
        elif command == "train":
            try:
                epochs = int(input("Введите количество эпох (по умолчанию 100): ") or 100)
                train_model(model, X, Y, epochs)
            except ValueError:
                print("Ошибка: Пожалуйста, введите целое число для количества эпох.")
        elif command == "use":
            if model is not None:
                use_model(model)
            else:
                print("Ошибка: Модель не загружена. Используйте команду 'load' для загрузки модели.")
        elif command == "show":
             show_data(X, Y)  # Передаём X и Y в функцию show_data
        elif command == "add":
             add_data(X, Y, data)
             # После добавления данных нужно обновить X и Y
             X, Y, data = load_data() # перезагружаем X, Y и data
        elif command == "edit":
            edit_data(X, Y, data)
            # После редактирования данных нужно обновить X и Y
            X, Y, data = load_data()  # перезагружаем X, Y и data
        elif command == "remove":
            remove_data(X, Y, data)
            # После удаления данных нужно обновить X и Y
            X, Y, data = load_data()  # перезагружаем X, Y и data
        elif command == "delete":
            delete_model()
        elif command == "quit":
            print("Выход из программы.")
            break
        else:
            print("Неизвестная команда. Используйте 'help' для просмотра списка команд.")
