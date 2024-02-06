#include "layer.h"
#include "fullyconnectedlayer.h"
#include "types.h"
#include "neuralnetwork.h"
#include "activationlayers.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <iterator>
#include "bitmap.h"

//Load flattened bitmap images from txt
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
        Vector current_line_vector_myType(current_line_vector.size(), 0);
        for(int i = 0; i < current_line_vector.size(); i++){
            current_line_vector_myType[i] = (myType)current_line_vector[i];
        }
        data.push_back(current_line_vector_myType);
    }

    std::cout << "Data loaded!" << std::endl;
    std::cout << "Data size: " << data.size() << std::endl << std::endl;
    return data;
}

//Suffle training data in pairs to preserver label/image order
void randomizeData(Matrix &input, Matrix &input2){
    std::random_device rd;
    std::mt19937 g(42);
    auto g1 = g;
    std::shuffle(input.begin(), input.end(), g);
    std::shuffle(input2.begin(), input2.end(), g1);
}

void normalizeData(Matrix &raw_data){
    for(int row = 0; row < raw_data.size(); row++){
        for(int col = 1; col < raw_data[0].size(); col++){
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

//Sava bmp images
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

//Extract the labels from the loaded txt
Matrix extractLabels(Matrix &input){
    Matrix labels(input.size(), Vector(0,0));

    for(int row = 0; row < input.size(); row++){
        labels[row].push_back(input[row].front());
        input[row].erase(input[row].begin());
    }
    return labels;
}

int main(){

    const int image_width = 28;
    const int image_height = 28;

    Matrix latent;
    Matrix generated;
    CE loss;

    NeuralNetwork neurons({
        std::make_shared<FullyConnectedLayer>(784, 32),
        std::make_shared<ElementWiseActivationLayer>(leakyRelu, leakyReluDerivative),
        std::make_shared<FullyConnectedLayer>(32, 10),
        std::make_shared<ElementWiseActivationLayer>(leakyRelu, leakyReluDerivative),
        std::make_shared<SoftmaxActivationLayer>(),
    });

    Matrix data = loadData("./mnist_train_githubsmall.txt");
    Matrix test = loadData("mnist_test.txt");
    Matrix labels = createOneHotMatrix(extractLabels(data), 10);
    Matrix test_labels = createOneHotMatrix(extractLabels(test), 10);
    Matrix batch;
    Matrix labels_batch;
    normalizeData(data);
    normalizeData(test);

    int epochs = 10;
    int batch_size = Layer::batch_size;
    int batches_per_epoch = data.size() / batch_size;

    for(int i = 0; i < epochs; i++){

        for(int batch_idx = 0; batch_idx < batches_per_epoch; batch_idx++){

            batch = calculateBatches(data,batch_size,batch_idx);
            labels_batch = calculateBatches(labels,batch_size,batch_idx);


            // Forward-propagation
            latent = neurons.forward(batch);

            if(batch_idx % 20 == 0){
                std::cout << loss.forward(latent, labels_batch) << std::endl;
                std::cout << "Current batch: " << batch_idx << "/" << batches_per_epoch << " of epoch: " << i+1 << "/" << epochs << std::endl;
                SaveBatch(batch, image_width, image_height, batch_size, "images/", "real");
            }
            Matrix derivative_wrt_output = loss.backward(latent, labels_batch);
            // Back-propagation and optimization
            neurons.backward(derivative_wrt_output);
        }
        randomizeData(data, labels);
    }
    
    int correct_counter = 0;
    std::cout << "////   Starting inference   ////" << std::endl;
    for(int i = 0; i < test.size(); i++){
        Matrix inference = neurons.forward(Matrix(1, test[i]));

        auto max_elem_inference = std::max_element(inference[0].begin(), inference[0].end());
        size_t inference_index = std::distance(inference[0].begin(), max_elem_inference);

        auto max_elem_expected = std::max_element(test_labels[i].begin(), test_labels[i].end());
        size_t test_index = std::distance(test_labels[i].begin(), max_elem_expected);

        if(inference_index == test_index){
            correct_counter++;
        }
        std::cout << "Infered value: " << inference_index << " with a confidence of: " << *max_elem_inference*100 << "%" << "///Expected value: " << test_index << "\n";

    }
    std::cout << "Preccission of the model: " << ((myType)correct_counter/test.size())*100 << "%\n";
}
