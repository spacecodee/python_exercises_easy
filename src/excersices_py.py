import numpy as np
import seaborn as sns
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as mplt
import matplotlib.animation as animation


# Línea de regresión con ruído aleatorio (tendencia lineal)
# Una línea de regresión nos permite predecir el valor de una variable dependiente (y)
# a partir de una variable independiente (x).
def linear_tendency():
    points = 50  # Número de puntos
    generator = np.random.default_rng(0)  # Para reproducibilidad
    x = generator.standard_normal(points)  # puntos aleatorios en 1D
    tendency = generator.standard_normal(points)  # Ruido (tendencia lineal)
    y = 1 + 2 * x + tendency  # Línea de regresión

    # Paso 2: Calcular la línea de regresión
    x_mean = np.mean(x)  # Media de x
    y_mean = np.mean(y)  # Media de y

    numerator = np.sum((x - x_mean) * (y - y_mean))  # Numerador de la pendiente el cual significa la covarianza
    print(numerator)
    denominator = np.sum((x - x_mean) ** 2)  # Denominador de la pendiente el cual significa la varianza
    print(denominator)

    b1 = numerator / denominator  # Pendiente de la línea de regresión
    b0 = y_mean - (b1 * x_mean)  # Intersección en y de la línea de regresión

    # Paso 3: Visualizar los datos y la línea de regresión
    mplt.scatter(x, y, color='green')  # Datos originales
    mplt.plot(x, b0 + b1 * x, color='blue')  # Línea de regresión
    mplt.title('Regresión Lineal')
    mplt.xlabel('x')
    mplt.ylabel('y')
    mplt.show()


# Se desea crear una gráfica de correlación para visualizar la relación entre dos variables, las cuales son aleatorias.
# Nota: Una gráfica de correlación es una representación visual de la relación entre dos variables.
def correlation_matrix():
    # Crear un DataFrame de ejemplo
    np.random.seed(0)
    df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
    # Calcular la matriz de correlación
    corr_matrix = df.corr()
    print(corr_matrix)

    # Crear una máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Crear una figura de matplotlib
    _, _ = mplt.subplots(figsize=(15, 10))

    # Dibujar el mapa de calor con la máscara
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    mplt.show()


# show the world map with the countries and the map without the countries
def show_countries_map():
    # Cargar el conjunto de datos incorporado de geopandas que contiene las geometrías de todos los países del mundo
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df = pd.DataFrame({
        'country': ['France', 'Germany', 'China', 'Brazil', 'Peru', 'Spain'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    # Fusiona el DataFrame con el GeoDataFrame en la columna 'country'
    merged = world.set_index('name').join(df.set_index('country'))

    # Dibuja el mapa del mundo con los valores de la columna 'value'
    merged.plot(column='value', legend=True)

    # Dibujar el mapa del mundo sin los paises
    world.plot()

    mplt.show()


def world_population_bar():
    # Paso 1: Crear un DataFrame con los nombres de los países y su población
    data = {
        'Country': ['China', 'India', 'USA', 'Indonesia', 'Pakistan'],
        'Population': [1393000000, 1366000000, 331000000, 273000000, 225000000]
    }
    df = pd.DataFrame(data)

    # Paso 2: Crear un gráfico de barras con matplotlib
    mplt.bar(df['Country'], df['Population'])
    mplt.title('Población por país')
    mplt.xlabel('País')
    mplt.ylabel('Población')
    mplt.show()


if __name__ == '__main__':
    linear_tendency()
    correlation_matrix()
    show_countries_map()
    world_population_bar()
