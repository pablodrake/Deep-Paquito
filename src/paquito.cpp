#include "layer.h"
#include "fullyconnectedlayer.h"
#include "types.h"
#include "neuralnetwork.h"
#include "elementwiseactivationlayer.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <iterator>
#include "bitmap.h"

Matrix loadData(const char* file_name){
    std::cout << "Loading " << file_name << "\n";

    std::ifstream file(file_name, std::ios::in);

    std::string line;
    size_t line_count = 0;
    while(std::getline(file, line)){
        line_count++;
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    Matrix data;
    data.reserve(line_count);

    while(std::getline(file, line)){
        std::istringstream is(line);
        std::vector<int> current_line_vector = (std::vector<int>(std::istream_iterator<int>(is),
                                      std::istream_iterator<int>()));
        Vector current_line_vector_myType(current_line_vector.size()-1, 0);
        for(int i = 0; i < current_line_vector.size()-1; i++){
            current_line_vector_myType[i] = (myType)current_line_vector[i+1];
        }
        data.push_back(current_line_vector_myType);
    }

    std::cout << "Data loaded!" << std::endl;
    std::cout << "Data size: " << data.size() << std::endl << std::endl;
    return data;
}

void randomizeData(Matrix &input){

    std::random_device rd;
    std::mt19937 g(42);
    std::shuffle(input.begin(), input.end(), g);
}

void normalizeData(Matrix &raw_data){
    for(int row = 0; row < raw_data.size(); row++){
        for(int col = 0; col < raw_data[0].size(); col++){
            raw_data[row][col]/=255;
        }
    }
}

Matrix calculateBatches(const Matrix &input, int batch_size, int batch_index){
    Matrix output;

    for(int row = batch_size * batch_index; row < batch_size * (batch_index+1); row++){
        output.push_back(input[row]);
    }
    return output;
}

void SaveBatch(const Matrix &batch, int width, int height, int image_number, std::string path, std::string name){

    auto image = reserveSpaceImage(height, width);

    for(int image_idx = 0; image_idx < image_number; image_idx++){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                image[i][j][0] = (unsigned char)(int)(batch[image_idx][i*width+j] * 255);
                image[i][j][1] = (unsigned char)(int)(batch[image_idx][i*width+j] * 255);
                image[i][j][2] = (unsigned char)(int)(batch[image_idx][i*width+j] * 255);

            }
        }

        std::string image_name = path + "/" + std::to_string(image_idx) + "_" + name + ".bmp";
        generateBitmapImage(image, height, width, (char*)image_name.c_str());
    }

    freeSpaceImage(image, height, width);
}


int main(){

    const int image_width = 28;
    const int image_height = 28;

    Matrix latent;
    Matrix generated;
    MSE loss;

    NeuralNetwork encoder({
        std::make_shared<FullyConnectedLayer>(784, 128),
        std::make_shared<ElementWiseActivationLayer>(leakyRelu, leakyReluDerivative),
        std::make_shared<FullyConnectedLayer>(128, 64),
        std::make_shared<ElementWiseActivationLayer>(leakyRelu, leakyReluDerivative),
        std::make_shared<FullyConnectedLayer>(64, 16),
        std::make_shared<ElementWiseActivationLayer>(leakyRelu, leakyReluDerivative),
    });
    NeuralNetwork decoder({
        std::make_shared<FullyConnectedLayer>(16, 64),
        std::make_shared<ElementWiseActivationLayer>(leakyRelu, leakyReluDerivative),
        std::make_shared<FullyConnectedLayer>(64, 128),
        std::make_shared<ElementWiseActivationLayer>(leakyRelu, leakyReluDerivative),
        std::make_shared<FullyConnectedLayer>(128, 784),
        std::make_shared<ElementWiseActivationLayer>(sigmoid, sigmoidDerivative),
    });

    Matrix data = loadData("./mnist_train_githubsmall.txt");
    Matrix batch;
    normalizeData(data);

    int epochs = 1;
    int batch_size = Layer::batch_size;
    int batches_per_epoch = data.size() / batch_size;

    Matrix test(1, Vector(image_width*image_height, (double)rand()/RAND_MAX));
    for(int i = 0; i < epochs; i++){
        randomizeData(batch);
        for(int batch_idx = 0; batch_idx < batches_per_epoch; batch_idx++){

            batch = calculateBatches(data,batch_size,batch_idx);
            std::cout << "Current batch: " << batch_idx << " of " << batches_per_epoch << " of epoch: " << i << std::endl;

            // Forward-propagation
            latent = encoder.forward(batch);
            generated = decoder.forward(latent);
            Matrix generate = {
                {0.05, 0.3, 0.4, 0.26, 0.3, 0.18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            };

            if(batch_idx % 20 == 0){
                std::cout << loss.forward(generated, batch) << std::endl;
                SaveBatch(batch, image_width, image_height, batch_size, "images/", "real");
                SaveBatch(generated, image_width, image_height, batch_size, "images/", "generated");
            }

            // Back-propagation and optimization
            Matrix derivative_wrt_output = loss.backward(generated, batch);
            encoder.backward(decoder.backward(derivative_wrt_output));
            SaveBatch(decoder.forward(generate), image_width, image_height, 1, "images/", "syntetik");
        }
    }
}
