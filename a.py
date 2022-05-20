# zadanie 1
# Tablica 1, bo można zrobić tablicę ze stringa
# Tablica 4 (ostatnia), bo z krotki pythonowej można stworzyć tablicę

#Zadanie 2
import matplotlib.pyplot as plt
import numpy as np

def liniowa(x, a: float, b: float):
    return a * x + b
x = np.arange(-5, 5, 0.1)
y = liniowa(x, -0.5, 1)
plt.plot(x, y)
plt.show()

#Zadanie 3
import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    return 1/(1.25 + np.cos(x))

x = np.arange(0, 6.28, 0.04)
y = f1(x)
y_list = list(y)
y_array = np.array(y)

plt.plot(x, y_list)
plt.show()

plt.plot(x, y_array)
plt.show()

print(y_list)
print(y_array)

# Tablica numpy dodatkowo formatuje swoją zawartość przed wyświetleniem jej na ekranie
# W przypadku wykresów nie ma żadnej różnicy 



# Zadanie 4
tab = np.random.random(6)
print('srednia', np.mean(tab))
print('suma', np.sum(tab))
print('odchylenie standardowe', np.std(tab))

tab12 = tab[[0, 1]]
tab135 = tab[[0, 2, 4]]
print('elementy 1, 2:', tab12)
print('elementy 1, 3, 5:', tab135)
