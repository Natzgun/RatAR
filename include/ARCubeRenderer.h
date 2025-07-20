#pragma once

// --- Inclusiones de Librerías ---
// GLAD: Para cargar los punteros de funciones de OpenGL. Debe incluirse antes que GLFW.
#include <glad/glad.h>
// GLFW: Para la gestión de la ventana y los eventos de entrada.
#include <GLFW/glfw3.h>
// GLM: Para las operaciones matemáticas de gráficos (vectores, matrices).
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
// OpenCV: Para el manejo de matrices y la conversión de vectores de pose.
#include <opencv2/opencv.hpp>
// Estándar de C++
#include <iostream>

// --- Definición de Shaders como Cadenas de Texto ---

// Shader de Vértices: Procesa cada vértice del modelo.
// - Transforma la posición del vértice al espacio de la pantalla usando las matrices MVP.
// - Pasa la posición del fragmento y la normal al Fragment Shader para el cálculo de la iluminación.
inline const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;

    out vec3 FragPos;
    out vec3 Normal;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

// Shader de Fragmentos: Calcula el color final de cada píxel del objeto.
// - Implementa el modelo de iluminación de Phong (ambiente, difusa, especular).
// - Esto le da al cubo una apariencia 3D realista con sombreado.
inline const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;
    in vec3 Normal;

    uniform vec3 objectColor;
    uniform vec3 lightColor;
    uniform vec3 lightPos;
    uniform vec3 viewPos;

    void main() {
        // Iluminación Ambiental
        float ambientStrength = 0.2;
        vec3 ambient = ambientStrength * lightColor;

        // Iluminación Difusa
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;

        // Iluminación Especular
        float specularStrength = 0.8;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;

        vec3 result = (ambient + diffuse + specular) * objectColor;
        FragColor = vec4(result, 1.0);
    }
)";

// Shader para dibujar la textura de fondo (el video de la cámara).
inline const char* backgroundVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
)";

inline const char* backgroundFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D backgroundTexture;
    void main() {
        FragColor = texture(backgroundTexture, TexCoord);
    }
)";


class ARCubeRenderer {
public:
    ARCubeRenderer() = default;
    ~ARCubeRenderer() { cleanup(); }

    // Inicializa GLFW, GLAD, la ventana, los shaders y los buffers del cubo.
    bool init(int width, int height, const std::string& title) {
        // --- 1. Inicialización de GLFW y creación de la ventana ---
        if (!glfwInit()) {
            std::cerr << "Error: No se pudo inicializar GLFW." << std::endl;
            return false;
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (!window) {
            std::cerr << "Error: No se pudo crear la ventana de GLFW." << std::endl;
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int w, int h) {
            glViewport(0, 0, w, h);
        });

        // --- 2. Inicialización de GLAD ---
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "Error: No se pudo inicializar GLAD." << std::endl;
            return false;
        }

        // --- 3. Compilación de Shaders ---
        cubeShaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
        backgroundShaderProgram = createShaderProgram(backgroundVertexShaderSource, backgroundFragmentShaderSource);
        
        if (cubeShaderProgram == 0 || backgroundShaderProgram == 0) return false;

        // --- 4. Configuración de la Geometría del Cubo (VBO y VAO) ---
        setupCube();
        setupBackground();

        // --- 5. Configuración del estado de OpenGL ---
        glEnable(GL_DEPTH_TEST); // Habilitar el test de profundidad para que el cubo se dibuje correctamente.
        
        // Configurar textura para el fondo
        glGenTextures(1, &backgroundTexture);
        glBindTexture(GL_TEXTURE_2D, backgroundTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        return true;
    }

    // Renderiza un fotograma.
    void render(const cv::Mat& frame, const cv::Vec3d& rvec, const cv::Vec3d& tvec, const cv::Mat& cameraMatrix) {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- 1. Dibujar el fotograma de la cámara como fondo ---
        drawBackground(frame);
        
        // Limpiar solo el buffer de profundidad para dibujar el objeto 3D sobre el fondo.
        glClear(GL_DEPTH_BUFFER_BIT);

        // // --- 2. Si no hay marcador, no se dibuja el cubo ---
        // if (rvec.empty() || tvec.empty()) {
        //     return; // No se detectó marcador, solo se muestra el video.
        // }

        // --- 3. Dibujar el cubo 3D ---
        glUseProgram(cubeShaderProgram);

        // --- 4. Construir y pasar las matrices de transformación a los shaders ---
        // Matriz de Proyección: Convierte la matriz de cámara de OpenCV a una de OpenGL.
        glm::mat4 projection = buildProjectionMatrix(cameraMatrix, frame.cols, frame.rows, 0.1f, 100.0f);
        
        // Matriz de Vista: Convierte los vectores de pose de OpenCV (rvec, tvec) a una matriz de vista de OpenGL.
        glm::mat4 view = buildViewMatrix(rvec, tvec);
        
        // Matriz de Modelo: Define la posición, rotación y escala del objeto en el mundo (en este caso, es simple).
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(0.05f)); // Escalar el cubo para que coincida con el tamaño del marcador

        glUniformMatrix4fv(glGetUniformLocation(cubeShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(cubeShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(cubeShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

        // --- 5. Configurar y pasar los uniformes de iluminación ---
        glUniform3f(glGetUniformLocation(cubeShaderProgram, "objectColor"), 0.2f, 0.5f, 0.8f); // Un color azulado
        glUniform3f(glGetUniformLocation(cubeShaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(cubeShaderProgram, "lightPos"), 0.5f, 0.5f, -0.5f); // Posición de la luz
        glUniform3f(glGetUniformLocation(cubeShaderProgram, "viewPos"), 0.0f, 0.0f, 0.0f); // La cámara está en el origen

        // --- 6. Dibujar el cubo ---
        glBindVertexArray(cubeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
    }

    // Devuelve si la ventana debe cerrarse.
    bool windowShouldClose() {
        return glfwWindowShouldClose(window);
    }

    // Intercambia los buffers de la ventana y procesa los eventos.
    void pollEventsAndSwapBuffers() {
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Libera todos los recursos.
    void cleanup() {
        glDeleteVertexArrays(1, &cubeVAO);
        glDeleteBuffers(1, &cubeVBO);
        glDeleteProgram(cubeShaderProgram);
        glDeleteVertexArrays(1, &backgroundVAO);
        glDeleteBuffers(1, &backgroundVBO);
        glDeleteProgram(backgroundShaderProgram);
        glDeleteTextures(1, &backgroundTexture);

        if (window) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }

private:
    GLFWwindow* window = nullptr;
    GLuint cubeShaderProgram = 0, backgroundShaderProgram = 0;
    GLuint cubeVAO = 0, cubeVBO = 0;
    GLuint backgroundVAO = 0, backgroundVBO = 0, backgroundTexture = 0;

    // Compila un shader y devuelve su ID.
    GLuint compileShader(GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Error en la compilación del shader: " << infoLog << std::endl;
            return 0;
        }
        return shader;
    }

    // Enlaza los shaders para crear un programa.
    GLuint createShaderProgram(const char* vsSource, const char* fsSource) {
        GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vsSource);
        GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fsSource);
        if (vertexShader == 0 || fragmentShader == 0) return 0;

        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        int success;
        char infoLog[512];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 512, nullptr, infoLog);
            std::cerr << "Error en el enlazado del programa de shaders: " << infoLog << std::endl;
            return 0;
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return program;
    }

    // Configura los buffers para el cubo.
    void setupCube() {
        float vertices[] = {
            // Posiciones           // Normales
            -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
            -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

            -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
             0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
            -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
            -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,

            -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

             0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
             0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

            -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
            -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
            -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,

            -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
            -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
        };

        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);

        glBindVertexArray(cubeVAO);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        // Atributo de posición
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // Atributo de normal
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }

    void setupBackground() {
        float quadVertices[] = { 
            // positions   // texture Coords
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };
        glGenVertexArrays(1, &backgroundVAO);
        glGenBuffers(1, &backgroundVBO);
        glBindVertexArray(backgroundVAO);
        glBindBuffer(GL_ARRAY_BUFFER, backgroundVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    }
    
    void drawBackground(const cv::Mat& frame) {
        glUseProgram(backgroundShaderProgram);
        
        cv::Mat flippedFrame;
        cv::flip(frame, flippedFrame, 0); // Voltear verticalmente para OpenGL
        
        glBindTexture(GL_TEXTURE_2D, backgroundTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, flippedFrame.cols, flippedFrame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, flippedFrame.data);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, backgroundTexture);
        glUniform1i(glGetUniformLocation(backgroundShaderProgram, "backgroundTexture"), 0);

        glBindVertexArray(backgroundVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }

    // Construye la matriz de proyección de OpenGL a partir de la matriz de cámara de OpenCV.
    glm::mat4 buildProjectionMatrix(const cv::Mat& cameraMatrix, int screen_width, int screen_height, float near, float far) {
        float fx = cameraMatrix.at<double>(0, 0);
        float fy = cameraMatrix.at<double>(1, 1);
        float cx = cameraMatrix.at<double>(0, 2);
        float cy = cameraMatrix.at<double>(1, 2);

        glm::mat4 projection = glm::mat4(0.0f);
        projection[0][0] = 2.0f * fx / screen_width;
        projection[1][1] = 2.0f * fy / screen_height;
        projection[2][0] = 1.0f - 2.0f * cx / screen_width;
        projection[2][1] = 2.0f * cy / screen_height - 1.0f;
        projection[2][2] = -(far + near) / (far - near);
        projection[2][3] = -1.0f;
        projection[3][2] = -2.0f * far * near / (far - near);
        return projection;
    }

    // Construye la matriz de vista de OpenGL a partir de los vectores de pose de OpenCV.
    glm::mat4 buildViewMatrix(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
        cv::Mat rot_mat;
        cv::Rodrigues(rvec, rot_mat); // Convierte el vector de rotación a una matriz de rotación

        glm::mat4 view_matrix;
        // Copia los datos de la matriz de rotación y el vector de traslación
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                view_matrix[j][i] = rot_mat.at<double>(i, j);
            }
        }
        view_matrix[3][0] = tvec[0];
        view_matrix[3][1] = tvec[1];
        view_matrix[3][2] = tvec[2];
        
        view_matrix[3][3] = 1.0f;

        // --- Conversión de Sistema de Coordenadas ---
        // OpenCV: Cámara mira hacia +Z, Y hacia abajo.
        // OpenGL: Cámara mira hacia -Z, Y hacia arriba.
        // Se necesita una transformación para alinear los sistemas.
        static const glm::mat4 cv_to_gl = glm::mat4(
            1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1
        );
        
        return cv_to_gl * view_matrix;
    }
};
