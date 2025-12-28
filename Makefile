CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

SRC = \
src/main.cpp \
src/io/csv_reader.cpp \
src/data/dataset.cpp \
src/preprocessing/imputer.cpp \
src/preprocessing/scaler.cpp \
src/distance/distance.cpp \
src/model/knn_classifier.cpp \
src/model/knn_regressor.cpp \
src/metrics/classification_metrics.cpp \
src/metrics/regression_metrics.cpp

OBJ = $(SRC:.cpp=.o)
BIN = knn

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)
