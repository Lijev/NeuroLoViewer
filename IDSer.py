import json
import numpy as np

def load_data(filepath):
    """Загружает данные из JSON файла."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        X = data['X']
        Y = data['Y']
        return X, Y
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return [], []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {filepath}")
        return [], []
    except KeyError as e:
        print(f"Error: Missing key in JSON: {e}")
        return [], []

def save_data(filepath, X, Y):
    """Сохраняет данные в JSON файл."""
    try:
        data = {'X': X, 'Y': Y}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving {e}")

def add_data(X, Y, new_x, new_y):
    """Добавляет данные в списки X и Y."""
    X.append(new_x)
    Y.append(new_y)
    print(f"Added: X={new_x}, Y={new_y}")
    return X, Y

def edit_data(X, Y, edit_x, new_y):
    """Изменяет Y для заданного X."""
    try:
        edit_x_np = np.array(edit_x)
        X_np = np.array(X)

        indices = np.where(np.all(X_np == edit_x_np, axis=1))[0]

        if indices.size == 0:
            print(f"Error: X={edit_x} not found in the dataset.")
            return X, Y

        index = indices[0]
        Y[index] = new_y
        print(f"Edited: X={edit_x}, Y={new_y}")
    except Exception as e:
        print(f"An error occurred during editing: {e}")
    return X, Y

def remove_data(X, Y, remove_x):
    """Удаляет данные для заданного X."""
    try:
        remove_x_np = np.array(remove_x)
        X_np = np.array(X)
        indices = np.where(np.all(X_np == remove_x_np, axis=1))[0]

        if indices.size == 0:
            print(f"Error: X={remove_x} not found in the dataset.")
            return X, Y

        index = indices[0]
        del X[index]
        del Y[index]
        print(f"Removed: X={remove_x}")
    except Exception as e:
        print(f"An error occurred during removal: {e}")
    return X, Y

def show_data(X, Y, show_x_values):
    """Показывает Y для заданных X."""
    for show_x in show_x_values:
        try:
            show_x_np = np.array(show_x)
            X_np = np.array(X)
            indices = np.where(np.all(X_np == show_x_np, axis=1))[0]

            if indices.size == 0:
                print(f"Error: X={show_x} not found in the dataset.")
                continue

            index = indices[0]
            print(f"X={show_x}, Y={Y[index]}")

        except Exception as e:
            print(f"An error occurred during showing: {e}")


def get_x_input():
    """Получает ввод для X с обработкой 'all'."""
    x1 = float(input("Enter X1: "))
    x2 = float(input("Enter X2: "))
    x3_input = input("Enter X3 (number or 'all'): ").lower()

    if x3_input == "all":
        x3_values = list(range(1, 11))  # Список от 1 до 10 включительно
    else:
        try:
            x3_values = [float(x3_input)]
        except ValueError:
            print("Invalid input for X3.  Please enter a number or 'all'.")
            return None

    x_values = []
    for x3 in x3_values:
        x_values.append([x1, x2, x3])
    return x_values


def main():
    """Основная функция для работы с данными."""
    filepath = 'IDS.json'
    X, Y = load_data(filepath)

    while True:
        command = input("Enter command (add, edit, remove, show, quit): ").lower()

        if command == 'add':
            x_values = get_x_input()
            if x_values is None:
                continue

            for x in x_values:
                y_input = eval(input(f"Enter Y for X={x} (as a list, e.g., [1.0, 2.0, 3.0, 4.0]): "))  # Get Y for each X
                if not isinstance(y_input, list) or len(y_input) != 4:
                    print("Invalid input for Y. Please enter a list of 4 numbers.")
                    continue
                X, Y = add_data(X, Y, x, y_input)  # Add the data point

        elif command == 'edit':
             x_values = get_x_input()
             if x_values is None:
                 continue
             if len(x_values) > 1:
                print("Editing only supports one X at a time. Please enter a specific X3 value.")
                continue
             y_input = eval(input("Enter new Y (as a list, e.g., [1.0, 2.0, 3.0, 4.0]): "))  # Use eval carefully!
             if not isinstance(y_input, list) or len(y_input) != 4:
                 print("Invalid input for Y. Please enter a list of 4 numbers.")
                 continue
             X, Y = edit_data(X, Y, x_values[0], y_input)

        elif command == 'remove':
            x_values = get_x_input()
            if x_values is None:
                continue
            for x in x_values:
                X, Y = remove_data(X, Y, x)

        elif command == 'show':
            x_values = get_x_input()
            if x_values is None:
                continue
            show_data(X, Y, x_values)

        elif command == 'quit':
            break

        else:
            print("Invalid command. Please enter add, edit, remove, show, or quit.")

        save_data(filepath, X, Y)

if __name__ == "__main__":
    main()
