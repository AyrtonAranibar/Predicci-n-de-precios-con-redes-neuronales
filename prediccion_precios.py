import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#propiedades = pd.read_csv('ar_properties.csv')

#Limpieza, mantener solo los campos necesarios
#columnas_a_mantener = ['ad_type', 'l1', 'l2', 'rooms', 'bedrooms', 'bathrooms',
                       #'surface_total', 'surface_covered','price','currency','price_period', 'property_type', 'operation_type']

#propiedades = propiedades[columnas_a_mantener]



#datos_limpios = propiedades
#datos_limpios.to_csv('datos_limpios.csv', index=False)

#Carga del nuevo set de datos
propiedades = pd.read_csv('datos_limpios.csv')

#eliminar registros que no interesan
propiedades = propiedades[propiedades['operation_type'] == 'Venta']

#eliminar cualquier  registro que tenga un campo vacio
propiedades = propiedades.dropna(axis=0, how='any')



# Calcula el rango intercuartil (IQR)
Q1 = propiedades['price'].quantile(0.05)
Q3 = propiedades['price'].quantile(0.95)
IQR = Q3 - Q1

# Define un límite para identificar outliers
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Filtra el DataFrame para eliminar los outliers
propiedades = propiedades[(propiedades['price'] >= lower_limit) & (propiedades['price'] <= upper_limit)]

print(len(propiedades))


#Separar variables categoricas a numeros 0=vacio 1=campo
propiedades = pd.get_dummies(propiedades, columns=['ad_type'], prefix=['ad_type'])
propiedades = pd.get_dummies(propiedades, columns=['l1'], prefix=['l1'])
propiedades = pd.get_dummies(propiedades, columns=['l2'], prefix=['l2'])
propiedades = pd.get_dummies(propiedades, columns=['currency'], prefix=['currency'])
propiedades = pd.get_dummies(propiedades, columns=['price_period'], prefix=['price_period'])
propiedades = pd.get_dummies(propiedades, columns=['property_type'], prefix=['property_type'])

#eliminar el campo venta por q se sobreentiende q todos son de ventas
propiedades = propiedades.drop('operation_type', axis=1)

#Convertir a punto flotante para entrenamiento
columnas_a_convertir = [
    'rooms', 'bedrooms', 'bathrooms', 'surface_total', 'surface_covered', 'price',
    'ad_type_Propiedad', 'l1_Argentina', 'l1_Brasil', 'l1_Estados Unidos', 'l1_Uruguay',
    'l2_Bs.As. G.B.A. Zona Norte', 'l2_Bs.As. G.B.A. Zona Oeste', 'l2_Bs.As. G.B.A. Zona Sur',
    'l2_Buenos Aires Costa Atlántica', 'l2_Buenos Aires Interior', 'l2_Canelones', 'l2_Capital Federal',
    'l2_Chaco', 'l2_Chubut', 'l2_Colonia', 'l2_Corrientes', 'l2_Córdoba', 'l2_Entre Ríos',
    'l2_Florida', 'l2_Jujuy', 'l2_La Pampa', 'l2_Maldonado', 'l2_Maryland', 'l2_Mendoza',
    'l2_Miami', 'l2_Michigan', 'l2_Misiones', 'l2_Montevideo', 'l2_Neuquén', 'l2_Rio Grande do Norte',
    'l2_Rocha', 'l2_Río Negro', 'l2_Salta', 'l2_San Juan', 'l2_San Luis', 'l2_Santa Catarina',
    'l2_Santa Cruz', 'l2_Santa Fe', 'l2_Santiago Del Estero', 'l2_Tierra Del Fuego', 'l2_Tucumán',
    'currency_ARS', 'currency_PEN', 'currency_USD', 'price_period_Mensual',
    'property_type_Casa', 'property_type_Casa de campo', 'property_type_Cochera',
    'property_type_Departamento', 'property_type_Depósito', 'property_type_Local comercial',
    'property_type_Lote', 'property_type_Oficina', 'property_type_Otro', 'property_type_PH'
]

for columna in columnas_a_convertir:
    propiedades[columna] = propiedades[columna].astype(float)

muestra = propiedades.head(150)
muestra.to_excel('muestra.xlsx', index=False)


# Divide los datos en conjuntos de entrenamiento y prueba

X = propiedades.drop('price', axis=1) 
y = propiedades['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Modelo

# Definir el modelo
# Definir las capas de entrada
entrada = tf.keras.layers.Input(shape=(60,))
capa1 = tf.keras.layers.Dense(units=60, activation='relu')(entrada)
capa2 = tf.keras.layers.Dense(units=60, activation='relu')(capa1)
capa3 = tf.keras.layers.Dense(units=60, activation='relu')(capa2)
salida = tf.keras.layers.Dense(units=1)(capa3)

# Crear el modelo funcional
modelo = tf.keras.Model(inputs=entrada, outputs=salida)

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_absolute_error'
)



#entrenamiento
print("Empezando entrenamiento")
historial = modelo.fit(X_train, y_train, epochs=50, batch_size=32)
print("modelo entrenado")




#visualizacion del aprendizaje
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()


# Realizar predicciones en datos de prueba
# Predicciones del modelo
predicciones = modelo.predict(X_test) 

# Calcular MAE, MSE y R^2
mae = mean_absolute_error(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print("Mean Absolute Error",mae)
print("Mean Squared Error",mse)
print("coeficiente de determinación",r2)

# Crear la gráfica de dispersión con puntos más pequeños (s)
plt.scatter(y_test, predicciones, s=10)  # Puedes ajustar el valor de 's' según tu preferencia
plt.xlabel('Valor Real (y_test)')
plt.ylabel('Predicciones (y_pred)')
plt.title('Comparación entre Valores Reales y Predicciones')

# Agregar la línea diagonal de igualdad
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=1)

plt.show()

# Crear un DataFrame con valores predichos y valores reales
resultados = pd.DataFrame({'Valor Real': y_test, 'Predicción': predicciones.flatten()})

# Calcular el porcentaje de error 
resultados['Porcentaje de Error (%)'] = (abs((resultados['Predicción'] - resultados['Valor Real']) / resultados['Valor Real']) * 100)
resultados.to_excel('resultados_predicciones.xlsx', index=False)

# Calcular el promedio del porcentaje de error
porcentaje_error_promedio = resultados['Porcentaje de Error (%)'].mean()

# Imprimir el promedio del porcentaje de error
print(f'El promedio del Porcentaje de Error (%) es: {porcentaje_error_promedio}')