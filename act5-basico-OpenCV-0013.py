import cv2
# Meredith Aguirre NC = 0013
print("-+-+-+- 1. Leer y mostrar la imagen -+-+-+-")
# Leer la imagen desde el archivo
image = cv2.imread('Michael_Jackson.jpg')

# Verificar que la imagen se haya cargado correctamente
if image is None:
    print("Error: No se pudo cargar 'Michael_Jackson.jpg'. Verifica que el archivo exista en la misma carpeta.")
    exit()

# Mostrar la imagen en una ventana
cv2.imshow('Imagen Original - Michael_Jackson', image)

# Esperar a que se presione una tecla y cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()


print("-+-+-+- 2. Convertir a escala de grises -+-+-+-")
# Convertir de BGR a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Guardar la imagen en escala de grises
cv2.imwrite('Michael_Jackson_gris.jpg', gray_image)


print("-+-+-+- 3. Redimensionar la imagen -+-+-+-")
# Redimensionar la imagen a 300x200 píxeles
resized_image = cv2.resize(image, (300, 200))

# Guardar la imagen redimensionada
cv2.imwrite('Michael_Jackson_redimensionada.jpg', resized_image)


print("-+-+-+- 4. Rotar la imagen 90 grados -+-+-+-")
# Obtener el alto y ancho de la imagen
(h, w) = image.shape[:2]

# Definir el centro de la imagen
center = (w // 2, h // 2)

# Crear la matriz de rotación (centro, ángulo, escala)
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)

# Aplicar la rotación
rotated_image = cv2.warpAffine(image, matrix, (w, h))

# Guardar la imagen rotada
cv2.imwrite('Michael_Jackson_rotada.jpg', rotated_image)


print("-+-+-+- 5. Desenfocar la imagen (Desenfoque Gaussiano) -+-+-+-")
# Aplicar desenfoque Gaussiano
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# Guardar la imagen desenfocada
cv2.imwrite('Michael_Jackson_desenfocada.jpg', blurred_image)


print("-+-+-+- 6. Detección de bordes con el algoritmo Canny -+-+-+-")
# Aplicar el detector de bordes Canny sobre la imagen en escala de grises
edges = cv2.Canny(gray_image, 100, 200)

# Guardar la imagen con los bordes detectados
cv2.imwrite('Michael_Jackson_bordes.jpg', edges)


print("-+-+-+- 7. Dibujar formas y agregar texto -+-+-+-")
# Hacer una copia de la imagen original para no modificarla directamente
image_shapes = image.copy()

# Dibujar un rectángulo (imagen, punto_inicial, punto_final, color_BGR, grosor)
cv2.rectangle(image_shapes, (50, 50), (200, 200), (255, 0, 0), 3)

# Agregar texto a la imagen
cv2.putText(image_shapes, 'OpenCV', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Guardar la imagen con formas y texto
cv2.imwrite('Michael_Jackson_formas_texto.jpg', image_shapes)


print("-+-+-+- 8. Umbralización (Thresholding binario) -+-+-+-")
# Aplicar umbralización binaria sobre la imagen en escala de grises
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Guardar la imagen umbralizada
cv2.imwrite('Michael_Jackson_umbral.jpg', thresh_image)


print("-+-+-+- 9. Detección de contornos -+-+-+-")
# Hacer otra copia de la imagen original para dibujar los contornos encima
image_contours = image.copy()

# Encontrar los contornos usando los bordes detectados previamente
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos sobre la copia de la imagen original
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

# Guardar la imagen final con los contornos
cv2.imwrite('Michael_Jackson_contornos.jpg', image_contours)

print("¡Listo! Todas las imágenes procesadas se han guardado en la carpeta.")
print("Hecho por Meredith Aguirre NC = 0013")