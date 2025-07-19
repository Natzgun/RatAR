#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/glm.hpp>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
};

class ModelLoader {
public:
    static bool loadOBJ(const std::string& filepath, 
                       std::vector<Vertex>& vertices, 
                       std::vector<unsigned int>& indices) {
        
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: No se pudo abrir el archivo PLY " << filepath << std::endl;
            return false;
        }
        
        std::string line;
        int vertexCount = 0;
        int faceCount = 0;
        bool headerEnded = false;
        
        // Leer header
        while (std::getline(file, line) && !headerEnded) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            
            if (token == "element") {
                std::string elementType;
                iss >> elementType;
                
                if (elementType == "vertex") {
                    iss >> vertexCount;
                } else if (elementType == "face") {
                    iss >> faceCount;
                }
            } else if (token == "end_header") {
                headerEnded = true;
            }
        }
        
        vertices.clear();
        indices.clear();
        
        // Leer vértices
        for (int i = 0; i < vertexCount; i++) {
            if (!std::getline(file, line)) break;
            
            std::istringstream iss(line);
            Vertex vertex;
            
            iss >> vertex.position.x >> vertex.position.y >> vertex.position.z;
            
            // Intentar leer normales si están disponibles
            if (!(iss >> vertex.normal.x >> vertex.normal.y >> vertex.normal.z)) {
                vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            
            vertex.texCoord = glm::vec2(0.0f, 0.0f);
            vertices.push_back(vertex);
        }
        
        // Leer caras
        for (int i = 0; i < faceCount; i++) {
            if (!std::getline(file, line)) break;
            
            std::istringstream iss(line);
            int numVertices;
            iss >> numVertices;
            
            if (numVertices >= 3) {
                std::vector<unsigned int> faceIndices;
                for (int j = 0; j < numVertices; j++) {
                    unsigned int index;
                    iss >> index;
                    faceIndices.push_back(index);
                }
                
                // Triangular cara si tiene más de 3 vértices
                for (int j = 1; j < numVertices - 1; j++) {
                    indices.push_back(faceIndices[0]);
                    indices.push_back(faceIndices[j]);
                    indices.push_back(faceIndices[j + 1]);
                }
            }
        }
        
        file.close();
        
        // Calcular normales si no existen
        calculateNormals(vertices, indices);
        
        std::cout << "Modelo PLY cargado: " << vertices.size() << " vertices, " 
                  << indices.size() << " indices" << std::endl;
        
        return true;
    }
    
    // Función para procesar vértices en formato OBJ
    static void processVertex(const std::string& vertexData,
                             std::vector<unsigned int>& positionIndices,
                             std::vector<unsigned int>& normalIndices,
                             std::vector<unsigned int>& texCoordIndices) {
        
        std::vector<std::string> parts = split(vertexData, '/');
        
        if (!parts.empty() && !parts[0].empty()) {
            positionIndices.push_back(std::stoi(parts[0]) - 1);
        }
        
        if (parts.size() > 1 && !parts[1].empty()) {
            texCoordIndices.push_back(std::stoi(parts[1]) - 1);
        }
        
        if (parts.size() > 2 && !parts[2].empty()) {
            normalIndices.push_back(std::stoi(parts[2]) - 1);
        }
    }
    
    // Función para dividir strings
    static std::vector<std::string> split(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        
        return tokens;
    }
    
    // Calcular normales para modelos que no las tienen
    static void calculateNormals(std::vector<Vertex>& vertices, 
                                const std::vector<unsigned int>& indices) {
        
        // Inicializar normales a cero
        for (auto& vertex : vertices) {
            vertex.normal = glm::vec3(0.0f);
        }
        
        // Calcular normales por cara
        for (size_t i = 0; i < indices.size(); i += 3) {
            if (i + 2 < indices.size()) {
                unsigned int i1 = indices[i];
                unsigned int i2 = indices[i + 1];
                unsigned int i3 = indices[i + 2];
                
                if (i1 < vertices.size() && i2 < vertices.size() && i3 < vertices.size()) {
                    glm::vec3 v1 = vertices[i2].position - vertices[i1].position;
                    glm::vec3 v2 = vertices[i3].position - vertices[i1].position;
                    glm::vec3 normal = glm::normalize(glm::cross(v1, v2));
                    
                    vertices[i1].normal += normal;
                    vertices[i2].normal += normal;
                    vertices[i3].normal += normal;
                }
            }
        }
        
        // Normalizar todas las normales
        for (auto& vertex : vertices) {
            vertex.normal = glm::normalize(vertex.normal);
        }
    }
    
    // Generar coordenadas de textura esféricas
    static void generateSphericalUV(std::vector<Vertex>& vertices) {
        for (auto& vertex : vertices) {
            glm::vec3 pos = glm::normalize(vertex.position);
            
            vertex.texCoord.x = 0.5f + atan2(pos.z, pos.x) / (2.0f * M_PI);
            vertex.texCoord.y = 0.5f + asin(pos.y) / M_PI;
        }
    }
    
    // Centrar y escalar modelo
    static void normalizeModel(std::vector<Vertex>& vertices, float targetSize = 1.0f) {
        if (vertices.empty()) return;
        
        // Encontrar bounding box
        glm::vec3 minPos = vertices[0].position;
        glm::vec3 maxPos = vertices[0].position;
        
        for (const auto& vertex : vertices) {
            minPos = glm::min(minPos, vertex.position);
            maxPos = glm::max(maxPos, vertex.position);
        }
        
        // Calcular centro y escala
        glm::vec3 center = (minPos + maxPos) * 0.5f;
        glm::vec3 size = maxPos - minPos;
        float maxDimension = glm::max(glm::max(size.x, size.y), size.z);
        float scale = targetSize / maxDimension;
        
        // Aplicar transformaciones
        for (auto& vertex : vertices) {
            vertex.position = (vertex.position - center) * scale;
        }
        
        std::cout << "Modelo normalizado: centro=" << center.x << "," << center.y << "," << center.z 
                  << ", escala=" << scale << std::endl;
    }
    
    // Crear un cubo procedural
    static void createCube(std::vector<Vertex>& vertices, 
                          std::vector<unsigned int>& indices,
                          float size = 1.0f) {
        vertices.clear();
        indices.clear();
        
        float s = size * 0.5f;
        
        // Vértices del cubo
        vertices = {
            // Cara frontal
            {{-s, -s,  s}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{ s, -s,  s}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
            {{ s,  s,  s}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
            {{-s,  s,  s}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
            
            // Cara trasera
            {{-s, -s, -s}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
            {{ s, -s, -s}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{ s,  s, -s}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
            {{-s,  s, -s}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
            
            // Cara izquierda
            {{-s, -s, -s}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-s, -s,  s}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{-s,  s,  s}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            {{-s,  s, -s}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            
            // Cara derecha
            {{ s, -s, -s}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{ s, -s,  s}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{ s,  s,  s}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            {{ s,  s, -s}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            
            // Cara superior
            {{-s,  s,  s}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            {{ s,  s,  s}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{ s,  s, -s}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{-s,  s, -s}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            
            // Cara inferior
            {{-s, -s,  s}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{ s, -s,  s}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{ s, -s, -s}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-s, -s, -s}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}}
        };
        
        indices = {
            0, 1, 2, 2, 3, 0,       // Frontal
            4, 5, 6, 6, 7, 4,       // Trasera
            8, 9, 10, 10, 11, 8,    // Izquierda
            12, 13, 14, 14, 15, 12, // Derecha
            16, 17, 18, 18, 19, 16, // Superior
            20, 21, 22, 22, 23, 20  // Inferior
        };
    }
    
    // Crear una esfera procedural
    static void createSphere(std::vector<Vertex>& vertices, 
                            std::vector<unsigned int>& indices,
                            float radius = 1.0f, 
                            int segments = 32) {
        vertices.clear();
        indices.clear();
        
        // Generar vértices
        for (int lat = 0; lat <= segments; lat++) {
            float theta = lat * M_PI / segments;
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);
            
            for (int lon = 0; lon <= segments; lon++) {
                float phi = lon * 2 * M_PI / segments;
                float sinPhi = sin(phi);
                float cosPhi = cos(phi);
                
                Vertex vertex;
                vertex.position.x = radius * sinTheta * cosPhi;
                vertex.position.y = radius * cosTheta;
                vertex.position.z = radius * sinTheta * sinPhi;
                
                vertex.normal = glm::normalize(vertex.position);
                
                vertex.texCoord.x = (float)lon / segments;
                vertex.texCoord.y = (float)lat / segments;
                
                vertices.push_back(vertex);
            }
        }
        
        // Generar índices
        for (int lat = 0; lat < segments; lat++) {
            for (int lon = 0; lon < segments; lon++) {
                int current = lat * (segments + 1) + lon;
                int next = current + segments + 1;
                
                indices.push_back(current);
                indices.push_back(next);
                indices.push_back(current + 1);
                
                indices.push_back(current + 1);
                indices.push_back(next);
                indices.push_back(next + 1);
            }
        }
    }
};

#endif // MODEL_LOADER_H.is_open()) {
