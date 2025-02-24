import json
import numpy as np

def load_data(filepath):
    """Loads data from a JSON file. Returns X and Y, or empty lists if an error occurs."""
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
    """Saves data to a JSON file, creating the file if it doesn't exist."""
    try:
        data = {'X': X, 'Y': Y}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)  # Use indent for readability
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving: {e}")

def add_data(X, Y, new_x, new_y):
    """Appends new data points to the existing X and Y lists."""
    X.append(new_x)
    Y.append(new_y)
    print(f"Added: X={new_x}, Y={new_y}")
    return X, Y

def edit_data(X, Y, edit_x, new_y):
    """
    Edits the Y value for a given X in the dataset.  Handles potential errors if X isn't found.
    Uses NumPy for efficient searching.
    """
    try:
        edit_x_np = np.array(edit_x)
        X_np = np.array(X)

        indices = np.where(np.all(X_np == edit_x_np, axis=1))[0] # Find indices where the whole row matches

        if indices.size == 0:
            print(f"Error: X={edit_x} not found in the dataset.")
            return X, Y  # Important:  Return original data if no match is found

        index = indices[0] # Use only first match

        Y[index] = new_y
        print(f"Edited: X={edit_x}, Y={new_y}")

    except Exception as e:
        print(f"An error occurred during editing: {e}") # Generic error message is helpful for debugging

    return X, Y

def remove_data(X, Y, remove_x):
    """Removes a data point (X and corresponding Y) from the dataset."""
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
    """Displays the Y value for each provided X value if it exists in the dataset."""
    for show_x in show_x_values:
        try:
            show_x_np = np.array(show_x)
            X_np = np.array(X)
            indices = np.where(np.all(X_np == show_x_np, axis=1))[0]

            if indices.size == 0:
                print(f"Error: X={show_x} not found in the dataset.")
                continue # go to the next item

            index = indices[0]
            print(f"X={show_x}, Y={Y[index]}")

        except Exception as e:
            print(f"An error occurred during showing: {e}")


def get_x_input():
    """Prompts the user for X1, X2, and X3 input, handling the 'all' option for X3."""
    x1 = float(input("Enter X1: "))
    x2 = float(input("Enter X2: "))
    x3_input = input("Enter X3 (number or 'all'): ").lower()

    if x3_input == "all":
        x3_values = list(range(1, 11))  # List from 1 to 10 inclusive
    else:
        try:
            x3_values = [float(x3_input)]
        except ValueError:
            print("Invalid input for X3. Please enter a number or 'all'.")
            return None

    # Create a list of X values for all combinations
    x_values = []
    for x3 in x3_values:
        x_values.append([x1, x2, x3]) # creates a list of x values using list comprehension.
    return x_values # returns the x value

def main():
    """Main function to interact with the user and manage data operations."""
    filepath = 'IDS.json'
    X, Y = load_data(filepath)

    while True:
        command = input("Enter command (add, edit, remove, show, quit): ").lower()

        if command == 'add':
            x_values = get_x_input() # Gets x values
            if x_values is None: # if x_values are invalid
                continue # Goes to start of loop (skips current iteration)

            for x in x_values:
                y_input = eval(input(f"Enter Y for X={x} (as a list, e.g., [1.0, 2.0, 3.0, 4.0]): "))
                if not isinstance(y_input, list) or len(y_input) != 4:
                    print("Invalid input for Y. Please enter a list of 4 numbers.")
                    continue  # Skip to the next X value

                X, Y = add_data(X, Y, x, y_input) # Adds x value and y value to x and y

        elif command == 'edit':
            x_values = get_x_input()
            if x_values is None:
                continue
            if len(x_values) > 1:
                print("Editing only supports one X at a time. Please enter a specific X3 value.")
                continue  # Skip to the next X value

            y_input = eval(input("Enter new Y (as a list, e.g., [1.0, 2.0, 3.0, 4.0]): "))  # Use eval carefully!
            if not isinstance(y_input, list) or len(y_input) != 4:
                print("Invalid input for Y. Please enter a list of 4 numbers.")
                continue  # Skip to the next X value
            X, Y = edit_data(X, Y, x_values[0], y_input)

        elif command == 'remove':
            x_values = get_x_input() # gets x values from user input
            if x_values is None: # checks x values for valid input.
                continue # goes to start of loop

            for x in x_values: # Iterates through all x_values
                X, Y = remove_data(X, Y, x) # removes values of X and Y if match X and Y

        elif command == 'show':
            x_values = get_x_input() # Gets X values from user input
            if x_values is None: # if input for X is empty it will continue the loop.
                continue # goes to start of loop

            show_data(X, Y, x_values)  # Shows the data to the user

        elif command == 'quit':
            break

        else:
            print("Invalid command. Please enter add, edit, remove, show, or quit.") # If none of the commands get satisfied, this will run

        save_data(filepath, X, Y) # Saves the data to the file

if __name__ == "__main__":
    main()
