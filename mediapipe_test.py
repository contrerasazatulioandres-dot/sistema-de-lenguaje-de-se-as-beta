import cv2
import mediapipe as mp

# Diccionario de señas LSE (Lengua de Señas Española)
SENIAS_LSE = {
    # Señas básicas
    "mano_abierta": "Hola",
    "pulgar_arriba": "Bien",
    "seña_paz": "Paz",
    "seña_ok": "De acuerdo",
    "señalar": "Mira/Allí",
    "corazón": "Te quiero",
    # Números (1-10)
    "uno": "1",
    "dos": "2", 
    "tres": "3",
    "cuatro": "4",
    "cinco": "5",
    "seis": "6",
    "siete": "7",
    "ocho": "8",
    "nueve": "9",
    "diez": "10",
    # Letras
    "a": "A",
    "b": "B",
    "c": "C",
    # Frases comunes
    "gracias": "Gracias",
    "por_favor": "Por favor",
    "amor": "Amor"
}

# Configuración de MediaPipe Hands
mp_manos = mp.solutions.hands
manos = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Utilidades de dibujo
mp_dibujo = mp.solutions.drawing_utils

# Inicializar cámara
camara = cv2.VideoCapture(0)

def detectar_senia(landmarks_mano):
    """Detecta señas LSE basadas en los puntos de referencia de la mano"""
    landmarks = landmarks_mano.landmark
    
    # Estados de los dedos (1 = extendido, 0 = doblado)
    dedos = []
    for punta, base in [(8,6), (12,10), (16,14), (20,18)]:
        dedos.append(1 if landmarks[punta].y < landmarks[base].y else 0)
    
    pulgar = 1 if landmarks[4].x < landmarks[3].x else 0
    
    # Detección de señas mejorada
    if sum(dedos) == 4: return "mano_abierta"  # Todos los dedos extendidos
    if sum(dedos) == 0 and pulgar: return "pulgar_arriba"
    if dedos[0:2] == [1,1] and sum(dedos[2:]) == 0: return "seña_paz"
    if (abs(landmarks[4].x - landmarks[8].x) < 0.03 and 
        abs(landmarks[4].y - landmarks[8].y) < 0.03): return "seña_ok"
    if sum(dedos) == 1 and dedos[0] == 1: return "señalar"
    
    # Detección de números (1-10)
    if sum(dedos) == 1 and dedos[0] == 1: return "uno"
    if sum(dedos) == 2 and dedos[0:2] == [1,1]: return "dos" 
    if sum(dedos) == 3 and dedos[0:3] == [1,1,1]: return "tres"
    if sum(dedos) == 4 and dedos[3] == 0: return "cuatro"
    if sum(dedos) == 5: return "cinco"
    if (dedos[0] == 1 and dedos[1] == 0 and dedos[2] == 0 and 
        dedos[3] == 1 and pulgar == 0): return "seis"
    if (dedos[0] == 1 and dedos[1] == 1 and dedos[2] == 0 and
        dedos[3] == 1 and pulgar == 0): return "siete"
    if (dedos[0] == 1 and dedos[1] == 1 and dedos[2] == 1 and
        dedos[3] == 1 and pulgar == 0): return "ocho"
    if (dedos[0] == 0 and dedos[1] == 1 and dedos[2] == 1 and
        dedos[3] == 1 and pulgar == 0): return "nueve"
    if (dedos[0] == 1 and dedos[1] == 0 and dedos[2] == 0 and
        dedos[3] == 0 and pulgar == 1): return "diez"
    
    # Seña de corazón mejorada
    # Verificar que los dedos estén doblados (excepto pulgar e índice)
    fingers_bent = (landmarks[12].y > landmarks[10].y and  # Dedo medio
                   landmarks[16].y > landmarks[14].y and   # Anular
                   landmarks[20].y > landmarks[18].y)      # Meñique
    
    # Verificar que pulgar e índice formen un corazón (puntas cercanas)
    thumb_index_close = (abs(landmarks[4].x - landmarks[8].x) < 0.05 and 
                        abs(landmarks[4].y - landmarks[8].y) < 0.05)
    
    # Verificar que pulgar e índice estén extendidos
    thumb_extended = landmarks[4].y < landmarks[2].y
    index_extended = landmarks[8].y < landmarks[6].y
    
    if fingers_bent and thumb_index_close and thumb_extended and index_extended:
        # Dibujar línea entre pulgar e índice para visualización
        cv2.line(imagen, 
                (int(landmarks[4].x * imagen.shape[1]), int(landmarks[4].y * imagen.shape[0])),
                (int(landmarks[8].x * imagen.shape[1]), int(landmarks[8].y * imagen.shape[0])),
                (0, 0, 255), 2)
        return "corazón"
    
    return None

while camara.isOpened():
    exito, imagen = camara.read()
    if not exito:
        print("Ignorando fotograma vacío de la cámara.")
        continue
    
    # Procesamiento de imagen
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = manos.process(imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    
    # Crear fondo para texto
    overlay = imagen.copy()
    cv2.rectangle(overlay, (0,0), (imagen.shape[1], 80), (0,0,0), -1)
    imagen = cv2.addWeighted(overlay, 0.6, imagen, 0.4, 0)
    
    if resultados.multi_hand_landmarks:
        for landmarks_mano in resultados.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(
                imagen, landmarks_mano, mp_manos.HAND_CONNECTIONS)
            
            # Detectar y mostrar seña
            senia = detectar_senia(landmarks_mano)
            if senia:
                significado = SENIAS_LSE.get(senia, "Seña desconocida")
                
                # Mostrar significado de forma destacada
                cv2.putText(imagen, f"Seña: {senia}", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(imagen, f"Significado: {significado}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Feedback visual
                cv2.circle(imagen, 
                          (int(landmarks_mano.landmark[0].x * imagen.shape[1]),
                           int(landmarks_mano.landmark[0].y * imagen.shape[0])),
                          30, (0,255,0), 2)
    
    cv2.imshow('LSE - Reconocimiento de Señas', imagen)
    if cv2.waitKey(5) & 0xFF == 27:
        break

camara.release()
cv2.destroyAllWindows()
