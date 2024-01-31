SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INC_DIR = include


all: $(BIN_DIR)/paquito

$(BIN_DIR)/paquito: $(OBJ_DIR)/paquito.o $(OBJ_DIR)/layer.o $(OBJ_DIR)/fullyconnectedlayer.o $(OBJ_DIR)/math.o $(OBJ_DIR)/neuralnetwork.o $(OBJ_DIR)/loss.o $(OBJ_DIR)/elementwiseactivationlayer.o $(OBJ_DIR)/bitmap.o
	g++ -fopenmp -o $(BIN_DIR)/paquito $(OBJ_DIR)/layer.o $(OBJ_DIR)/fullyconnectedlayer.o $(OBJ_DIR)/math.o $(OBJ_DIR)/paquito.o $(OBJ_DIR)/neuralnetwork.o $(OBJ_DIR)/loss.o $(OBJ_DIR)/elementwiseactivationlayer.o $(OBJ_DIR)/bitmap.o -I$(INC_DIR)

$(OBJ_DIR)/paquito.o: $(SRC_DIR)/paquito.cpp
	g++ -fopenmp -o $(OBJ_DIR)/paquito.o -g -c $(SRC_DIR)/paquito.cpp -I$(INC_DIR)

$(OBJ_DIR)/layer.o: $(SRC_DIR)/layer.cpp $(INC_DIR)/layer.h
	g++ -fopenmp -o $(OBJ_DIR)/layer.o -g -c $(SRC_DIR)/layer.cpp -I$(INC_DIR)

$(OBJ_DIR)/fullyconnectedlayer.o: $(SRC_DIR)/fullyconnectedlayer.cpp $(INC_DIR)/fullyconnectedlayer.h
	g++ -fopenmp -o $(OBJ_DIR)/fullyconnectedlayer.o -g -c $(SRC_DIR)/fullyconnectedlayer.cpp -I$(INC_DIR)

$(OBJ_DIR)/neuralnetwork.o: $(SRC_DIR)/neuralnetwork.cpp $(INC_DIR)/neuralnetwork.h
	g++ -fopenmp -o $(OBJ_DIR)/neuralnetwork.o -g -c $(SRC_DIR)/neuralnetwork.cpp -I$(INC_DIR)

$(OBJ_DIR)/loss.o: $(SRC_DIR)/loss.cpp $(INC_DIR)/loss.h
	g++ -fopenmp -o $(OBJ_DIR)/loss.o -g -c $(SRC_DIR)/loss.cpp -I$(INC_DIR)

$(OBJ_DIR)/math.o: $(SRC_DIR)/math.cpp $(INC_DIR)/math.h
	g++ -fopenmp -o $(OBJ_DIR)/math.o -g -c $(SRC_DIR)/math.cpp -I$(INC_DIR)

$(OBJ_DIR)/elementwiseactivationlayer.o: $(SRC_DIR)/elementwiseactivationlayer.cpp $(INC_DIR)/elementwiseactivationlayer.h
	g++ -fopenmp -o $(OBJ_DIR)/elementwiseactivationlayer.o -g -c $(SRC_DIR)/elementwiseactivationlayer.cpp -I$(INC_DIR)
		
$(OBJ_DIR)/bitmap.o: $(SRC_DIR)/bitmap.cpp $(INC_DIR)/bitmap.h
	g++ -fopenmp -o $(OBJ_DIR)/bitmap.o -g -c $(SRC_DIR)/bitmap.cpp -I$(INC_DIR)

clean:
	rm -rf obj/*.o
	rm -rf bin/*
