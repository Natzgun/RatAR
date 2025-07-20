// main.cpp
//
// Título: Implementación de la Base para Realidad Aumentada con OpenCV y OpenGL
// Cumple con los requisitos del Trabajo Final de Computación Gráfica.
//
// Funcionalidades Implementadas:
// ... (mismos requisitos que antes, ahora con renderizado 3D)
// 5. [NUEVO] Renderizado de Objeto 3D con OpenGL:
//    - Se crea una ventana OpenGL usando GLFW.
//    - El video de la cámara se dibuja como textura de fondo.
//    - Un cubo 3D con iluminación Phong se renderiza sobre el marcador ArUco.
//    - La pose (rvec, tvec) de OpenCV se usa para posicionar el cubo.
//
// Versión de OpenCV: 4.9.0
// Dependencias Adicionales: GLFW, GLAD, GLM
// Compilación: g++ -o main main.cpp glad.c -lglfw -lGL `pkg-config --cflags --libs opencv4`
// (Nota: Debes tener los archivos de cabecera de glad, glfw, glm y linkear las librerías)
// (Nota 2: glad.c debe generarse desde el servicio web de GLAD y ponerse junto a main.cpp)

#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// --- INCLUIR EL RENDERIZADOR DE OPENGL ---
#include "ARCubeRenderer.h"

class AugmentedRealityApp {
private:
  cv::VideoCapture cap;
  cv::Mat cameraMatrix, distCoeffs;
  cv::aruco::Dictionary dictionary;
  cv::aruco::ArucoDetector detector;
  
  // --- INSTANCIA DEL RENDERIZADOR ---
  ARCubeRenderer renderer;

  bool isCalibrated = false;
  std::string calibrationFilePath = "calibration_data.yml";

  const cv::Size boardSize{9, 6};
  const float squareSize_m = 0.025f;
  const float markerLength_m = 0.05f;

public:
  AugmentedRealityApp();
  ~AugmentedRealityApp();
  void run();

private:
  bool loadCalibration();
  void saveCalibration();
  void performCalibration();
  // Modificado para pasar los vectores de pose al renderizador
  void detectAndRender(cv::Mat &frame); 
  bool detectHandGesture(const cv::Mat &inputFrame);
};

AugmentedRealityApp::AugmentedRealityApp()
    : dictionary(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)),
      detector(dictionary) {
  cap.open(0);
  if (!cap.isOpened()) {
    std::cerr << "FATAL: No se pudo abrir la cámara." << std::endl;
    exit(-1);
  }
  isCalibrated = loadCalibration();
}

AugmentedRealityApp::~AugmentedRealityApp() {
  if (cap.isOpened())
    cap.release();
  // La limpieza de la ventana ahora la hace el destructor de ARCubeRenderer
  std::cout << "Aplicación finalizada." << std::endl;
}

bool AugmentedRealityApp::loadCalibration() {
  cv::FileStorage fs(calibrationFilePath, cv::FileStorage::READ);
  if (!fs.isOpened()) return false;
  fs["cameraMatrix"] >> cameraMatrix;
  fs["distCoeffs"] >> distCoeffs;
  fs.release();
  std::cout << "Calibración cargada exitosamente." << std::endl;
  return !cameraMatrix.empty() && !distCoeffs.empty();
}

void AugmentedRealityApp::saveCalibration() {
  cv::FileStorage fs(calibrationFilePath, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    std::cerr << "Error: No se pudo guardar el archivo de calibración." << std::endl;
    return;
  }
  fs << "cameraMatrix" << cameraMatrix;
  fs << "distCoeffs" << distCoeffs;
  fs.release();
  std::cout << "Calibración guardada." << std::endl;
}

// La calibración ahora muestra la ventana de OpenCV temporalmente
void AugmentedRealityApp::performCalibration() {
    std::cout << "\n--- INICIANDO PROCESO DE CALIBRACION ---" << std::endl;
    std::cout << "Muestre un tablero de ajedrez de 9x6 a la cámara." << std::endl;
    std::cout << "Presione 'c' para capturar una vista. Necesita 20 vistas buenas." << std::endl;

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
        if (frame.empty()) continue;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, boardSize, corners,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            cv::drawChessboardCorners(frame, boardSize, corners, found);
        }

        std::string msg = "Vistas: " + std::to_string(imagePoints.size()) + "/" + std::to_string(requiredImages);
        cv::putText(frame, msg, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Calibracion de Camara", frame); // Ventana temporal de OpenCV

        char key = (char)cv::waitKey(20);
        if (key == 'q') return;
        if (key == 'c' && found) {
            imagePoints.push_back(corners);
            objectPoints.push_back(objp);
            std::cout << "Vista " << imagePoints.size() << " capturada." << std::endl;
        }
    }

    cv::destroyWindow("Calibracion de Camara");
    std::cout << "Calculando parametros de la camara..." << std::endl;

    cv::Mat rvecs, tvecs;
    cv::calibrateCamera(objectPoints, imagePoints, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

    std::cout << "Calibracion completada." << std::endl;
    isCalibrated = true;
    saveCalibration();
}

// Procesa cada fotograma y lo pasa al renderizador.
void AugmentedRealityApp::detectAndRender(cv::Mat &frame) {
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners;
  detector.detectMarkers(frame, markerCorners, markerIds);

  cv::Vec3d rvec, tvec; // Vectores para pasar al renderizador

  if (!markerIds.empty()) {
    // Usamos solo el primer marcador detectado para simplificar
    std::vector<cv::Point3f> objPoints = {
        cv::Point3f(-markerLength_m / 2.f, markerLength_m / 2.f, 0),
        cv::Point3f(markerLength_m / 2.f, markerLength_m / 2.f, 0),
        cv::Point3f(markerLength_m / 2.f, -markerLength_m / 2.f, 0),
        cv::Point3f(-markerLength_m / 2.f, -markerLength_m / 2.f, 0)};

    // Estimar la pose del primer marcador
    cv::solvePnP(objPoints, markerCorners[0], cameraMatrix, distCoeffs, rvec, tvec);
    
    // Para depuración, podemos dibujar los ejes en el frame original
    cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, markerLength_m * 0.7f, 3);
  }

  bool gestureTrigger = detectHandGesture(frame);
  if (gestureTrigger) {
    cv::putText(frame, "GESTO DETECTADO!", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    // Aquí se podría cambiar el estado del objeto 3D, por ejemplo, su color o animación.
    // Por ahora, solo imprimimos en consola.
    std::cout << "Disparador de Gesto: ON" << std::endl;
  }

  // --- LLAMADA AL RENDERIZADOR DE OPENGL ---
  // Pasamos el frame, los vectores de pose y la matriz de la cámara.
  // El renderizador se encargará de dibujar el fondo y el cubo si hay pose.
  renderer.render(frame, rvec, tvec, cameraMatrix);
}

// Bucle principal de la aplicación.
void AugmentedRealityApp::run() {
  if (!isCalibrated) {
    performCalibration();
    if (!isCalibrated) {
      std::cerr << "La aplicación no puede continuar sin calibración. Saliendo." << std::endl;
      return;
    }
  }

  // --- INICIALIZAR EL RENDERIZADOR DE OPENGL ---
  cv::Mat tempFrame;
  cap >> tempFrame;
  if(!renderer.init(tempFrame.cols, tempFrame.rows, "Proyecto Final AR - OpenGL")) {
      std::cerr << "Fallo al inicializar el renderizador de OpenGL." << std::endl;
      return;
  }

  std::cout << "\n--- INICIANDO DETECCION ---" << std::endl;
  std::cout << "Apunte la camara a un marcador ArUco." << std::endl;
  std::cout << "Cierre la ventana para salir." << std::endl;

  cv::Mat frame;
  // El bucle ahora es controlado por la ventana de GLFW
  while (!renderer.windowShouldClose()) {
    cap >> frame;
    if (frame.empty()) break;
    
    detectAndRender(frame);
    
    renderer.pollEventsAndSwapBuffers();
  }
}

bool AugmentedRealityApp::detectHandGesture(const cv::Mat &inputFrame) {
    cv::Mat hsvFrame, skinMask;
    cv::cvtColor(inputFrame, hsvFrame, cv::COLOR_BGR2HSV);
    cv::inRange(hsvFrame, cv::Scalar(0, 48, 80), cv::Scalar(20, 255, 255), skinMask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(skinMask, skinMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    int maxAreaIdx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }

    if (maxAreaIdx != -1 && maxArea > 8000) {
        std::vector<int> hullIndices;
        cv::convexHull(contours[maxAreaIdx], hullIndices);
        if (hullIndices.size() > 3) {
            std::vector<cv::Vec4i> defects;
            cv::convexityDefects(contours[maxAreaIdx], hullIndices, defects);
            int fingerCount = 0;
            for (const cv::Vec4i &v : defects) {
                if (v[3] / 256.0 > 15) {
                    fingerCount++;
                }
            }
            if (fingerCount >= 3) {
                return true;
            }
        }
    }
    return false;
}

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
