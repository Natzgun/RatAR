#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

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

struct Model {
    GLuint vao = 0;
    GLuint vbo = 0;
    int vertexCount = 0;
    glm::vec3 diffuseColor = glm::vec3(0.8f, 0.8f, 0.8f);
};


class ARObjectRenderer {
public:
    ARObjectRenderer() = default;
    ~ARObjectRenderer() { cleanup(); }

    bool init(int width, int height, const std::string& title) {
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

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "Error: No se pudo inicializar GLAD." << std::endl;
            return false;
        }

        objectShaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
        backgroundShaderProgram = createShaderProgram(backgroundVertexShaderSource, backgroundFragmentShaderSource);
        
        if (objectShaderProgram == 0 || backgroundShaderProgram == 0) return false;

        setupBackground();

        glEnable(GL_DEPTH_TEST);
        
        glGenTextures(1, &backgroundTexture);
        glBindTexture(GL_TEXTURE_2D, backgroundTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        return true;
    }

    bool loadModel(const std::string& objPath, const std::string& mtlBasePath) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objPath.c_str(), mtlBasePath.c_str())) {
            std::cerr << "Error al cargar el modelo OBJ: " << warn << err << std::endl;
            return false;
        }
        if (!warn.empty()) {
            std::cout << "Advertencia de TinyObjLoader: " << warn << std::endl;
        }

        std::vector<float> vertices;
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                vertices.push_back(attrib.vertices[3 * index.vertex_index + 0]);
                vertices.push_back(attrib.vertices[3 * index.vertex_index + 1]);
                vertices.push_back(attrib.vertices[3 * index.vertex_index + 2]);

                if (index.normal_index >= 0) {
                    vertices.push_back(attrib.normals[3 * index.normal_index + 0]);
                    vertices.push_back(attrib.normals[3 * index.normal_index + 1]);
                    vertices.push_back(attrib.normals[3 * index.normal_index + 2]);
                } else {
                    vertices.push_back(0.0f);
                    vertices.push_back(0.0f);
                    vertices.push_back(0.0f);
                }
            }
        }
        
        if (!materials.empty()) {
            loadedModel.diffuseColor.r = materials[0].diffuse[0];
            loadedModel.diffuseColor.g = materials[0].diffuse[1];
            loadedModel.diffuseColor.b = materials[0].diffuse[2];
        }

        loadedModel.vertexCount = vertices.size() / 6;

        glGenVertexArrays(1, &loadedModel.vao);
        glGenBuffers(1, &loadedModel.vbo);

        glBindVertexArray(loadedModel.vao);
        glBindBuffer(GL_ARRAY_BUFFER, loadedModel.vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindVertexArray(0);

        std::cout << "Modelo cargado exitosamente: " << objPath << std::endl;
        std::cout << "Vértices procesados: " << loadedModel.vertexCount << std::endl;

        return true;
    }

    void render(const cv::Mat& frame, const cv::Vec3d& rvec, const cv::Vec3d& tvec, const cv::Mat& cameraMatrix) {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        drawBackground(frame);
        
        glClear(GL_DEPTH_BUFFER_BIT);

        // --- 2. Dibujar el objeto 3D si se ha cargado un modelo y es visible ---
        if (loadedModel.vao != 0 && tvec[2] > 0) { // tvec[2] > 0 comprueba si el marcador está frente a la cámara
            glUseProgram(objectShaderProgram);

            glm::mat4 projection = buildProjectionMatrix(cameraMatrix, frame.cols, frame.rows, 0.1f, 100.0f);
            glm::mat4 view = buildViewMatrix(rvec, tvec);
            glm::mat4 model = glm::mat4(1.0f);

            glm::mat4 animationTransform = getAnimationTransform();
            model = model * animationTransform;

            model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::scale(model, glm::vec3(0.001f));

            glUniformMatrix4fv(glGetUniformLocation(objectShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
            glUniformMatrix4fv(glGetUniformLocation(objectShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(objectShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

            glUniform3fv(glGetUniformLocation(objectShaderProgram, "objectColor"), 1, glm::value_ptr(loadedModel.diffuseColor));
            glUniform3f(glGetUniformLocation(objectShaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);
            glUniform3f(glGetUniformLocation(objectShaderProgram, "lightPos"), 0.5f, 0.5f, -0.5f);
            glUniform3f(glGetUniformLocation(objectShaderProgram, "viewPos"), 0.0f, 0.0f, 0.0f);

            glBindVertexArray(loadedModel.vao);
            glDrawArrays(GL_TRIANGLES, 0, loadedModel.vertexCount);
            glBindVertexArray(0);
        }
    }

    bool windowShouldClose() {
        return glfwWindowShouldClose(window);
    }

    void pollEventsAndSwapBuffers() {
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    void triggerAnimation() {
        if (!animationActive) {
            animationActive = true;
            animationStartTime = glfwGetTime();
        }
    }
    void cleanup() {
        glDeleteVertexArrays(1, &loadedModel.vao);
        glDeleteBuffers(1, &loadedModel.vbo);
        
        glDeleteProgram(objectShaderProgram);
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
    GLuint objectShaderProgram = 0, backgroundShaderProgram = 0;
    GLuint backgroundVAO = 0, backgroundVBO = 0, backgroundTexture = 0;
    
    Model loadedModel;

    bool animationActive = false;
    float animationStartTime = 0.0f;
    const float animationDuration = 1.0f; 
    const float animationHeight = 0.05f;
    glm::mat4 getAnimationTransform() {
        if (animationActive) {
            float elapsedTime = glfwGetTime() - animationStartTime;

            if (elapsedTime < animationDuration) {
                float height = animationHeight * (elapsedTime / animationDuration);
                return glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, height, 0.0f));
            } else {
                animationActive = false;
            }
        }
        return glm::mat4(1.0f);
    }

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

    void setupBackground() {
        float quadVertices[] = { 
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
        cv::flip(frame, flippedFrame, 0);
        
        glBindTexture(GL_TEXTURE_2D, backgroundTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, flippedFrame.cols, flippedFrame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, flippedFrame.data);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, backgroundTexture);
        glUniform1i(glGetUniformLocation(backgroundShaderProgram, "backgroundTexture"), 0);

        glBindVertexArray(backgroundVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }

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

    glm::mat4 buildViewMatrix(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
        cv::Mat rot_mat;
        cv::Rodrigues(rvec, rot_mat);

        glm::mat4 view_matrix = glm::mat4(1.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                view_matrix[j][i] = rot_mat.at<double>(i, j);
            }
        }
        view_matrix[3][0] = tvec[0];
        view_matrix[3][1] = tvec[1];
        view_matrix[3][2] = tvec[2];
        
        static const glm::mat4 cv_to_gl = glm::mat4(
            1,  0,  0, 0,
            0, -1,  0, 0,
            0,  0, -1, 0,
            0,  0,  0, 1
        );
        
        return cv_to_gl * view_matrix;
    }
};
