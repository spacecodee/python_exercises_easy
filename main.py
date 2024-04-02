# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def get_prom():
    number1 = int(input("Enter the first number: "))
    number2 = int(input("Enter the second number: "))
    number3 = int(input("Enter the third number: "))

    return (0.2 * number1) + (0.3 * number2) + (0.5 * number3)


# Ejercicio 1: Escriba una función en Python que tome una lista de números y devuelva la suma de todos los números en la lista.
def sum_number_list(list_numbers):
    return sum(list_numbers)


# Ejercicio 2: Escriba una función en Python que tome una cadena de texto y devuelva un diccionario donde las claves son los caracteres en la cadena de texto y los valores son la cantidad de veces que cada caracter aparece en la cadena de texto.
def count_characters(cadena):
    dictionary = {}
    for character in cadena:
        if character in dictionary:
            dictionary[character] += 1
        else:
            dictionary[character] = 1
    return dictionary


# Ejercicio 3: Escriba una función en Python que tome dos números y devuelva su producto. Si el producto es mayor a 1000, entonces devuelva su suma.
def cal_sum_or_product(num1, num2):
    product = num1 * num2
    if product > 1000:
        return num1 + num2
    else:
        return product


# Ejercicio 4: Escriba una función en Python que tome una cadena de texto y devuelva la misma cadena pero con las palabras en orden inverso.
def invert_words(cadena):
    words = cadena.split()
    words_invert = words[::-1]
    word_invert = ' '.join(words_invert)
    return word_invert


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sum_number_list([1, 2, 3, 4, 5])
    print(f"The sum of the list is: ", sum_number_list([1, 2, 3, 4, 5]))
    print(count_characters("Hey, I am a python Programmer"))
    print("The product or sum: ", cal_sum_or_product(10, 20))
    print("The product or sum: ", cal_sum_or_product(80, 30))
    print("Invert: ", invert_words("Hey, I am a python Programmer"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
