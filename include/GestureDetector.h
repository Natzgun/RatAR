#ifndef GESTURE_DETECTOR_H
#define GESTURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <chrono>
#include <iostream>

enum class GestureType {
    NONE,
    HAND_OPEN,
    HAND_CLOSED,
    POINTING,
    SWIPE_LEFT,
    SWIPE_RIGHT,
    SWIPE_UP,
    SWIPE_DOWN
};

struct GestureEvent {
    GestureType type;
    cv::Point2f position;
    std::chrono::steady_clock::time_point timestamp;
    float confidence;
};

class GestureDetector {
private:
    cv::HOGDescriptor hog;
    cv::CascadeClassifier handCascade;
    cv::Ptr<cv::BackgroundSubtractor> backgroundSubtractor;
    
    // Parámetros de detección
    cv::Scalar lowerSkin, upperSkin;
    int minContourArea;
    int maxContourArea;
    
    // Histórico de gestos para detección de movimiento
    std::vector<cv::Point2f> handPositions;
    std::vector<std::chrono::steady_clock::time_point> timestamps;
    const size_t maxHistorySize = 10;
    
    // Estado del detector
    bool isInitialized;
    GestureType lastGesture;
    std::chrono::steady_clock::time_point lastGestureTime;
    
public:
    GestureDetector() : 
        minContourArea(1000),
        maxContourArea(50000),
        isInitialized(false),
        lastGesture(GestureType::NONE) {
        
        // Configurar detector HOG
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        
        // Configurar rango de color de piel (HSV)
        lowerSkin = cv::Scalar(0, 20, 70);
        upperSkin = cv::Scalar(20, 255, 255);
        
        // Crear substractor de fondo
        backgroundSubtractor = cv::createBackgroundSubtractorMOG2(500, 50, true);
    }
    
    bool initialize(const std::string& cascadePath = "") {
        // Cargar clasificador de manos si está disponible
        if (!cascadePath.empty()) {
            if (handCascade.load(cascadePath)) {
                std::cout << "Clasificador de manos cargado exitosamente" << std::endl;
            } else {
                std::cout << "Advertencia: No se pudo cargar el clasificador de manos" << std::endl;
            }
        }
        
        isInitialized = true;
        return true;
    }
    
    std::vector<GestureEvent> detectGestures(const cv::Mat& frame) {
        std::vector<GestureEvent> gestures;
        
        if (!isInitialized || frame.empty()) {
            return gestures;
        }
        
        // Detectar manos por color de piel
        auto handRegions = detectHandsBySkin(frame);
        
        // Detectar manos con clasificador si está disponible
        if (!handCascade.empty()) {
            auto cascadeHands = detectHandsByCascade(frame);
            handRegions.insert(handRegions.end(), cascadeHands.begin(), cascadeHands.end());
        }
        
        // Procesar cada región de mano detectada
        for (const auto& handRect : handRegions) {
            cv::Mat handROI = frame(handRect);
            
            // Análisis de la forma de la mano
            GestureType gestureType = analyzeHandShape(handROI);
            
            // Crear evento de gesto
            GestureEvent gesture;
            gesture.type = gestureType;
            gesture.position = cv::Point2f(handRect.x + handRect.width / 2.0f, 
                                         handRect.y + handRect.height / 2.0f);
            gesture.timestamp = std::chrono::steady_clock::now();
            gesture.confidence = 0.8f; // Valor base, se puede mejorar
            
            gestures.push_back(gesture);
            
            // Actualizar histórico para detección de movimiento
            updateHandHistory(gesture.position);
        }
        
        // Detectar gestos de movimiento
        auto movementGestures = detectMovementGestures();
        gestures.insert(gestures.end(), movementGestures.begin(), movementGestures.end());
        
        return gestures;
    }
    
    void drawGestures(cv::Mat& frame, const std::vector<GestureEvent>& gestures) {
        for (const auto& gesture : gestures) {
            cv::Point center(static_cast<int>(gesture.position.x), 
                           static_cast<int>(gesture.position.y));
            
            // Dibujar círculo en la posición del gesto
            cv::circle(frame, center, 20, cv::Scalar(0, 255, 0), 3);
            
            // Dibujar texto del tipo de gesto
            std::string gestureText = gestureTypeToString(gesture.type);
            cv::putText(frame, gestureText, 
                       cv::Point(center.x - 50, center.y - 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Dibujar confianza
            std::string confidenceText = "Conf: " + std::to_string(gesture.confidence).substr(0, 4);
            cv::putText(frame, confidenceText, 
                       cv::Point(center.x - 50, center.y + 50),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        }
        
        // Dibujar trayectoria de la mano
        drawHandTrajectory(frame);
    }
    
private:
    std::vector<cv::Rect> detectHandsBySkin(const cv::Mat& frame) {
        std::vector<cv::Rect> handRegions;
        
        // Convertir a HSV
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        
        // Aplicar filtro de color de piel
        cv::Mat skinMask;
        cv::inRange(hsv, lowerSkin, upperSkin, skinMask);
        
        // Morfología para limpiar la máscara
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(skinMask, skinMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, kernel);
        
        // Encontrar contornos
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Filtrar contornos por tamaño
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > minContourArea && area < maxContourArea) {
                cv::Rect boundingRect = cv::boundingRect(contour);
                
                // Verificar aspect ratio para filtrar manos
                double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
                if (aspectRatio > 0.5 && aspectRatio < 2.0) {
                    handRegions.push_back(boundingRect);
                }
            }
        }
        
        return handRegions;
    }
    
    std::vector<cv::Rect> detectHandsByCascade(const cv::Mat& frame) {
        std::vector<cv::Rect> hands;
        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        handCascade.detectMultiScale(gray, hands, 1.1, 3, 0, cv::Size(30, 30));
        
        return hands;
    }
    
    GestureType analyzeHandShape(const cv::Mat& handROI) {
        // Convertir a escala de grises
        cv::Mat gray;
        cv::cvtColor(handROI, gray, cv::COLOR_BGR2GRAY);
        
        // Aplicar threshold
        cv::Mat binary;
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        
        // Encontrar contornos
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) {
            return GestureType::NONE;
        }
        
        // Encontrar el contorno más grande
        auto largestContour = *std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        
        // Análisis de convexidad
        std::vector<cv::Point> hull;
        cv::convexHull(largestContour, hull);
        
        std::vector<int> hullIndices;
        cv::convexHull(largestContour, hullIndices);
        
        std::vector<cv::Vec4i> defects;
        if (hullIndices.size() > 3) {
            cv::convexityDefects(largestContour, hullIndices, defects);
        }
        
        // Clasificar gesto basado en defectos de convexidad
        int significantDefects = 0;
        for (const auto& defect : defects) {
            float depth = defect[3] / 256.0f;
            if (depth > 20) { // Umbral para defectos significativos
                significantDefects++;
            }
        }
        
        // Clasificación simple basada en defectos
        if (significantDefects >= 4) {
            return GestureType::HAND_OPEN;
        } else if (significantDefects <= 1) {
            return GestureType::HAND_CLOSED;
        } else if (significantDefects == 2) {
            return GestureType::POINTING;
        }
        
        return GestureType::NONE;
    }
    
    std::vector<GestureEvent> detectMovementGestures() {
        std::vector<GestureEvent> movementGestures;
        
        if (handPositions.size() < 3) {
            return movementGestures;
        }
        
        // Calcular vector de movimiento
        cv::Point2f startPos = handPositions[0];
        cv::Point2f endPos = handPositions.back();
        cv::Point2f movement = endPos - startPos;
        
        float distance = cv::norm(movement);
        
        // Verificar si hay movimiento significativo
        if (distance > 50) { // Umbral de distancia
            auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(
                timestamps.back() - timestamps[0]).count();
            
            if (timeDiff < 1000) { // Debe ocurrir en menos de 1 segundo
                GestureType gestureType = GestureType::NONE;
                
                // Determinar dirección del movimiento
                if (abs(movement.x) > abs(movement.y)) {
                    // Movimiento horizontal
                    gestureType = (movement.x > 0) ? GestureType::SWIPE_RIGHT : GestureType::SWIPE_LEFT;
                } else {
                    // Movimiento vertical
                    gestureType = (movement.y > 0) ? GestureType::SWIPE_DOWN : GestureType::SWIPE_UP;
                }
                
                GestureEvent gesture;
                gesture.type = gestureType;
                gesture.position = endPos;
                gesture.timestamp = std::chrono::steady_clock::now();
                gesture.confidence = std::min(1.0f, distance / 100.0f);
                
                movementGestures.push_back(gesture);
                
                // Limpiar histórico después de detectar movimiento
                handPositions.clear();
                timestamps.clear();
            }
        }
        
        return movementGestures;
    }
    
    void updateHandHistory(const cv::Point2f& position) {
        handPositions.push_back(position);
        timestamps.push_back(std::chrono::steady_clock::now());
        
        // Mantener solo el histórico reciente
        if (handPositions.size() > maxHistorySize) {
            handPositions.erase(handPositions.begin());
            timestamps.erase(timestamps.begin());
        }
    }
    
    void drawHandTrajectory(cv::Mat& frame) {
        if (handPositions.size() < 2) return;
        
        for (size_t i = 1; i < handPositions.size(); i++) {
            cv::Point p1(static_cast<int>(handPositions[i-1].x), 
                        static_cast<int>(handPositions[i-1].y));
            cv::Point p2(static_cast<int>(handPositions[i].x), 
                        static_cast<int>(handPositions[i].y));
            
            cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 2);
        }
    }
    
    std::string gestureTypeToString(GestureType type) {
        switch (type) {
            case GestureType::HAND_OPEN: return "ABIERTA";
            case GestureType::HAND_CLOSED: return "CERRADA";
            case GestureType::POINTING: return "SEÑALANDO";
            case GestureType::SWIPE_LEFT: return "IZQUIERDA";
            case GestureType::SWIPE_RIGHT: return "DERECHA";
            case GestureType::SWIPE_UP: return "ARRIBA";
            case GestureType::SWIPE_DOWN: return "ABAJO";
            default: return "NINGUNO";
        }
    }
};

#endif // GESTURE_DETECTOR_H
