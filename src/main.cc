// main.cpp
//
// Título: Implementación de la Base para Realidad Aumentada con OpenCV
// Cumple con los requisitos del Trabajo Final de Computación Gráfica.
//
// Funcionalidades Implementadas:
// 1.  [Requisito] Lenguaje de Programación: C++
// 2.  [Requisito] Uso de Patrón:
//     - Tablero de Ajedrez: Para una calibración precisa y robusta de la
//     cámara.
//     - Marcador ArUco ("otro"): Para la detección y seguimiento en tiempo
//     real.
// 3.  [Requisito] Estimación de Pose 3D para Video:
//     - Calcula los vectores 'rvec' (rotación) y 'tvec' (traslación) en cada
//     frame.
//     - Estos vectores son la salida directa para el módulo de renderizado de
//     OpenGL.
//     - Dibuja ejes 3D como prueba visual de que la pose es correcta.
// 4.  [Requisito] Reconocimiento de Gestos para Interacción:
//     - Incluye la función 'detectHandGesture' que devuelve 'true' al detectar
//     una mano abierta.
//     - Esta señal booleana es el disparador para la interacción con el objeto
//     3D.
//
// Versión de OpenCV: 4.9.0
// Compilación: g++ -o main main.cpp `pkg-config --cflags --libs opencv4`

#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <ARCubeRenderer.h>
#include <vector>

// --- Definición de la Clase Principal de la Aplicación ---

class AugmentedRealityApp {
private:
  // --- Miembros de la Clase ---
  cv::VideoCapture cap;
  cv::Mat cameraMatrix, distCoeffs;
  cv::aruco::Dictionary dictionary;
  cv::aruco::ArucoDetector detector;

  bool isCalibrated = false;
  std::string calibrationFilePath = "calibration_data.yml";

  // Parámetros para la calibración con tablero de ajedrez
  const cv::Size boardSize{9, 6}; // Esquinas internas: 9 de ancho, 6 de alto
  const float squareSize_m =
      0.025f; // Tamaño del cuadrado del tablero en metros (2.5 cm)

  // Parámetros para el marcador ArUco
  const float markerLength_m = 0.05f; // Tamaño del marcador en metros (5 cm)

public:
  AugmentedRealityApp();
  ~AugmentedRealityApp();
  void run(); // Método principal que ejecuta la aplicación

private:
  // --- Métodos Privados para organizar la lógica ---
  bool loadCalibration();
  void saveCalibration();
  void performCalibration();
  void detectAndProcess(cv::Mat &frame);
  bool detectHandGesture(const cv::Mat &inputFrame, cv::Mat &debugFrame);
};

// --- Implementación de los Métodos de la Clase ---

AugmentedRealityApp::AugmentedRealityApp()
    : dictionary(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)),
      detector(dictionary) {
  cap.open(0); // Abrir la cámara por defecto
  if (!cap.isOpened()) {
    std::cerr << "FATAL: No se pudo abrir la cámara." << std::endl;
    exit(-1);
  }
  // Intentar cargar una calibración previa para no repetir el proceso
  isCalibrated = loadCalibration();
}

AugmentedRealityApp::~AugmentedRealityApp() {
  if (cap.isOpened())
    cap.release();
  cv::destroyAllWindows();
  std::cout << "Aplicación finalizada." << std::endl;
}

// Carga los datos de calibración desde un archivo para evitar recalibrar.
bool AugmentedRealityApp::loadCalibration() {
  cv::FileStorage fs(calibrationFilePath, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }
  fs["cameraMatrix"] >> cameraMatrix;
  fs["distCoeffs"] >> distCoeffs;
  fs.release();
  std::cout << "Calibración cargada exitosamente desde " << calibrationFilePath
            << std::endl;
  return !cameraMatrix.empty() && !distCoeffs.empty();
}

// Guarda los datos de calibración en un archivo.
void AugmentedRealityApp::saveCalibration() {
  cv::FileStorage fs(calibrationFilePath, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    std::cerr << "Error: No se pudo guardar el archivo de calibración."
              << std::endl;
    return;
  }
  fs << "cameraMatrix" << cameraMatrix;
  fs << "distCoeffs" << distCoeffs;
  fs.release();
  std::cout << "Calibración guardada en " << calibrationFilePath << std::endl;
}

// [CUMPLE REQUISITO: Patrón de Tablero de Ajedrez]
// Guía al usuario a través del proceso de calibración de la cámara.
void AugmentedRealityApp::performCalibration() {
  std::cout << "\n--- INICIANDO PROCESO DE CALIBRACION ---" << std::endl;
  std::cout << "Muestre un tablero de ajedrez de 9x6 a la cámara." << std::endl;
  std::cout
      << "Presione 'c' para capturar una vista. Necesita 20 vistas buenas."
      << std::endl;

  std::vector<std::vector<cv::Point3f>> objectPoints;
  std::vector<std::vector<cv::Point2f>> imagePoints;
  std::vector<cv::Point3f> objp;

  for (int i = 0; i < boardSize.height; ++i) {
    for (int j = 0; j < boardSize.width; ++j) {
      objp.push_back(cv::Point3f(j * squareSize_m, i * squareSize_m, 0));
    }
  }

  cv::Mat frame, gray;
  const int requiredImages = 20;

  while (imagePoints.size() < requiredImages) {
    cap >> frame;
    if (frame.empty())
      continue;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(gray, boardSize, corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH |
                                               cv::CALIB_CB_NORMALIZE_IMAGE);

    if (found) {
      cv::cornerSubPix(
          gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                           0.1));
      cv::drawChessboardCorners(frame, boardSize, corners, found);
    }

    std::string msg = "Vistas: " + std::to_string(imagePoints.size()) + "/" +
                      std::to_string(requiredImages);
    cv::putText(frame, msg, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
    cv::imshow("Calibracion de Camara", frame);

    char key = (char)cv::waitKey(20);
    if (key == 'q')
      return;
    if (key == 'c' && found) {
      imagePoints.push_back(corners);
      objectPoints.push_back(objp);
      std::cout << "Vista " << imagePoints.size() << " capturada." << std::endl;
    }
  }

  cv::destroyWindow("Calibracion de Camara");
  std::cout << "Calculando parametros de la camara..." << std::endl;

  cv::Mat rvecs, tvecs;
  double rms = cv::calibrateCamera(objectPoints, imagePoints, frame.size(),
                                   cameraMatrix, distCoeffs, rvecs, tvecs);

  std::cout << "Calibracion completada. Error RMS: " << rms << std::endl;
  isCalibrated = true;
  saveCalibration();
}

// [CUMPLE REQUISITO: Detección en Video, Pose 3D y Gesto]
// Procesa cada fotograma de la cámara para detectar marcadores y gestos.
void AugmentedRealityApp::detectAndProcess(cv::Mat &frame) {
  cv::Mat frameCopy;
  frame.copyTo(frameCopy);

  // --- 1. Detección de Marcador ArUco ---
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners;
  detector.detectMarkers(frame, markerCorners, markerIds);

  // --- 2. Estimación de Pose 3D ---
  if (!markerIds.empty()) {
    cv::aruco::drawDetectedMarkers(frameCopy, markerCorners, markerIds);

    std::vector<cv::Vec3d> rvecs, tvecs;
    // Estimar la pose para cada marcador detectado
    for (size_t i = 0; i < markerIds.size(); ++i) {
      std::vector<cv::Point3f> objPoints = {
          cv::Point3f(-markerLength_m / 2.f, markerLength_m / 2.f, 0),
          cv::Point3f(markerLength_m / 2.f, markerLength_m / 2.f, 0),
          cv::Point3f(markerLength_m / 2.f, -markerLength_m / 2.f, 0),
          cv::Point3f(-markerLength_m / 2.f, -markerLength_m / 2.f, 0)};
      cv::Vec3d rvec, tvec;
      cv::solvePnP(objPoints, markerCorners[i], cameraMatrix, distCoeffs, rvec,
                   tvec);
      rvecs.push_back(rvec);
      tvecs.push_back(tvec);

      // Dibuja los ejes como prueba visual. OpenGL usará rvec y tvec para
      // dibujar el objeto.
      cv::drawFrameAxes(frameCopy, cameraMatrix, distCoeffs, rvec, tvec,
                        markerLength_m * 0.7f, 3);

  ARRenderer::drawCube(rvec, tvec, cameraMatrix);
    }

    // --- SALIDA PARA OPENGL ---
    // Estos son los datos que tu equipo necesita.
    // Se pueden pasar a una función de renderizado de OpenGL.
    // Por ahora, los imprimimos en la consola para depuración.
    std::cout << "Pose 3D -> rvec: " << rvecs[0].t()
              << " | tvec: " << tvecs[0].t() << std::endl;
  }

  // --- 3. Reconocimiento de Gestos ---
  bool gestureTrigger = detectHandGesture(frame, frameCopy);
  if (gestureTrigger) {
    cv::putText(frameCopy, "GESTO DETECTADO!", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    // --- SEÑAL DE INTERACCIÓN PARA OPENGL ---
    // Aquí se llamaría a una función que active la animación del objeto 3D.
    // Ejemplo: openGL_animateObject(true);
    std::cout << "Disparador de Gesto: ON" << std::endl;
  }

  cv::imshow("Proyecto Final AR - OpenCV", frameCopy);
}

// Bucle principal de la aplicación.
void AugmentedRealityApp::run() {
  if (!isCalibrated) {
    performCalibration();
    if (!isCalibrated) {
      std::cerr << "La aplicación no puede continuar sin calibración. Saliendo."
                << std::endl;
      return;
    }
  }

  std::cout << "\n--- INICIANDO DETECCION ---" << std::endl;
  std::cout << "Apunte la camara a un marcador ArUco." << std::endl;
  std::cout << "Presione 'q' para salir." << std::endl;

ARRenderer::initOpenGL(640, 480);
  cv::Mat frame;
  while (true) {
    cap >> frame;
    if (frame.empty())
      break;
    detectAndProcess(frame);
    char key = (char)cv::waitKey(10);
    if (key == 'q' || key == 27)
      break;
  }
ARRenderer::terminateOpenGL();
}

// [CUMPLE REQUISITO: Reconocimiento de Gesto]
// Detecta una mano abierta segmentando por color de piel y analizando la forma.
bool AugmentedRealityApp::detectHandGesture(const cv::Mat &inputFrame,
                                            cv::Mat &debugFrame) {
  cv::Mat hsvFrame, skinMask;
  // Convertir a espacio de color HSV para una mejor segmentación de la piel
  cv::cvtColor(inputFrame, hsvFrame, cv::COLOR_BGR2HSV);
  // Rango de color de piel en HSV (puede necesitar ajustes)
  cv::inRange(hsvFrame, cv::Scalar(0, 48, 80), cv::Scalar(20, 255, 255),
              skinMask);

  // Limpiar la máscara para eliminar ruido
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
  cv::morphologyEx(skinMask, skinMask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(skinMask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // Encontrar el contorno más grande (asumimos que es la mano)
  double maxArea = 0;
  int maxAreaIdx = -1;
  for (size_t i = 0; i < contours.size(); i++) {
    double area = cv::contourArea(contours[i]);
    if (area > maxArea) {
      maxArea = area;
      maxAreaIdx = i;
    }
  }

  // Si se encontró un contorno suficientemente grande
  if (maxAreaIdx != -1 && maxArea > 8000) {
    std::vector<int> hullIndices;
    // Calcular el casco convexo (la figura que envuelve la mano)
    cv::convexHull(contours[maxAreaIdx], hullIndices);

    if (hullIndices.size() > 3) {
      std::vector<cv::Vec4i> defects;
      // Calcular los defectos de convexidad (los valles entre los dedos)
      cv::convexityDefects(contours[maxAreaIdx], hullIndices, defects);
      int fingerCount = 0;
      for (const cv::Vec4i &v : defects) {
        float depth = v[3] / 256.0; // Profundidad del valle
        if (depth > 15) { // Si el valle es suficientemente profundo, es un dedo
          fingerCount++;
        }
      }
      // Una mano abierta tiene 4 valles entre 5 dedos.
      if (fingerCount >= 3) { // Ser flexible (3, 4 o 5 dedos)
        return true;
      }
    }
  }
  return false;
}

// --- Función Principal que inicia todo ---
int main() {
  try {
    AugmentedRealityApp app;
    app.run();
  } catch (const cv::Exception &e) {
    std::cerr << "Error de OpenCV: " << e.what() << std::endl;
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}
