#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H
#include "layer.h"
#include "math.h"

class FullyConnectedLayer : public Layer{
    private:
        Matrix weights;
        Vector bias;
    
    public:
        FullyConnectedLayer(int inputLayers, int outputLayers);
        Matrix forward(const Matrix &output) override;
        /**
         * Entra la derivada de la funcion de entrada con respecto a la salida y sale la derivada de la funcion de costo con respecto a la entrada.
         * Restas los valores a los pesos para minimizar la funcion de coste 
        */
        Matrix backward(const Matrix &derivative_wrt_output) override;
        void coutLayer();

};

#endif //FULLYCONNECTED_LAYER_H