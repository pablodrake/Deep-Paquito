#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include "loss.h"
#include <memory>

class NeuralNetwork{
    private:
        std::vector<std::shared_ptr<Layer>> layers;
    public:
        NeuralNetwork(const std::vector<std::shared_ptr<Layer>> &hidden_layers);
        Matrix forward(const Matrix &input);
        Matrix backward(const Matrix &expected_output);
};

#endif //NEURALNETWORK_H