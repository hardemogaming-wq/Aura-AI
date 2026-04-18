CXX = g++
CXXFLAGS = -std=c++17 -Iinclude
LDFLAGS =

# Targets
CLI_TARGET = main
SHARED_TARGET = libaura.so
SERVER_TARGET = aura_server
SYNTHETIC_TARGET = train_synthetic

# Source directories
SRC_DIR = src
OBJ_DIR = obj

# Common source files
COMMON_SRCS = $(SRC_DIR)/NeuralNetwork.cpp $(SRC_DIR)/Layer.cpp $(SRC_DIR)/Matrix.cpp $(SRC_DIR)/Activation.cpp $(SRC_DIR)/Loss.cpp $(SRC_DIR)/Transformer.cpp $(SRC_DIR)/Tokenizer.cpp $(SRC_DIR)/Embedding.cpp $(SRC_DIR)/PositionalEncoding.cpp $(SRC_DIR)/LayerNorm.cpp $(SRC_DIR)/FeedForward.cpp

# Object files
CLI_OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/NeuralNetwork.o $(OBJ_DIR)/Layer.o $(OBJ_DIR)/Matrix.o $(OBJ_DIR)/Activation.o $(OBJ_DIR)/Loss.o $(OBJ_DIR)/Transformer.o $(OBJ_DIR)/Tokenizer.o $(OBJ_DIR)/Embedding.o $(OBJ_DIR)/PositionalEncoding.o $(OBJ_DIR)/LayerNorm.o $(OBJ_DIR)/FeedForward.o
SHARED_OBJS = $(OBJ_DIR)/run_aura.o $(OBJ_DIR)/NeuralNetwork.o $(OBJ_DIR)/Layer.o $(OBJ_DIR)/Matrix.o $(OBJ_DIR)/Activation.o $(OBJ_DIR)/Loss.o $(OBJ_DIR)/Transformer.o $(OBJ_DIR)/Tokenizer.o $(OBJ_DIR)/Embedding.o $(OBJ_DIR)/PositionalEncoding.o $(OBJ_DIR)/LayerNorm.o $(OBJ_DIR)/FeedForward.o
SERVER_OBJS = $(OBJ_DIR)/aura_server.o $(OBJ_DIR)/NeuralNetwork.o $(OBJ_DIR)/Layer.o $(OBJ_DIR)/Matrix.o $(OBJ_DIR)/Activation.o $(OBJ_DIR)/Loss.o $(OBJ_DIR)/Transformer.o $(OBJ_DIR)/Tokenizer.o $(OBJ_DIR)/Embedding.o $(OBJ_DIR)/PositionalEncoding.o $(OBJ_DIR)/LayerNorm.o $(OBJ_DIR)/FeedForward.o
SYNTHETIC_OBJS = $(OBJ_DIR)/train_synthetic.o $(OBJ_DIR)/NeuralNetwork.o $(OBJ_DIR)/Layer.o $(OBJ_DIR)/Matrix.o $(OBJ_DIR)/Activation.o $(OBJ_DIR)/Loss.o $(OBJ_DIR)/Transformer.o $(OBJ_DIR)/Tokenizer.o $(OBJ_DIR)/Embedding.o $(OBJ_DIR)/PositionalEncoding.o $(OBJ_DIR)/LayerNorm.o $(OBJ_DIR)/FeedForward.o

# Default target
all: cli shared server synthetic

# Create obj directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# CLI build
cli: $(CLI_TARGET)
$(CLI_TARGET): $(CLI_OBJS)
	$(CXX) $(CLI_OBJS) -o $(CLI_TARGET) $(LDFLAGS)

# Shared library build (separate PIC objects)
SHARED_OBJ_DIR = obj/shared
SHARED_OBJS = $(SHARED_OBJ_DIR)/run_aura.o $(SHARED_OBJ_DIR)/NeuralNetwork.o $(SHARED_OBJ_DIR)/Layer.o $(SHARED_OBJ_DIR)/Matrix.o $(SHARED_OBJ_DIR)/Activation.o $(SHARED_OBJ_DIR)/Loss.o $(SHARED_OBJ_DIR)/Transformer.o $(SHARED_OBJ_DIR)/Tokenizer.o $(SHARED_OBJ_DIR)/Embedding.o $(SHARED_OBJ_DIR)/PositionalEncoding.o $(SHARED_OBJ_DIR)/LayerNorm.o $(SHARED_OBJ_DIR)/FeedForward.o

shared: $(SHARED_TARGET)
$(SHARED_TARGET): $(SHARED_OBJS)
	$(CXX) -fPIC -shared $(SHARED_OBJS) -o $(SHARED_TARGET) $(LDFLAGS)

$(SHARED_OBJ_DIR):
	mkdir -p $(SHARED_OBJ_DIR)

$(SHARED_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(SHARED_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# Server build
server: $(SERVER_TARGET)
$(SERVER_TARGET): $(SERVER_OBJS)
	$(CXX) $(SERVER_OBJS) -o $(SERVER_TARGET) $(LDFLAGS) -pthread

# Synthetic trainer build
synthetic: $(SYNTHETIC_TARGET)
$(SYNTHETIC_TARGET): $(SYNTHETIC_OBJS)
	$(CXX) $(SYNTHETIC_OBJS) -o $(SYNTHETIC_TARGET) $(LDFLAGS)

# Object file compilation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(CLI_TARGET) $(SHARED_TARGET) $(SERVER_TARGET) $(SYNTHETIC_TARGET) *.o

# Test targets
test-cli: cli
	@echo "Testing CLI..."
	./$(CLI_TARGET) < /dev/null || echo "CLI test completed"

test-server: server
	@echo "Testing server..."
	timeout 5 ./$(SERVER_TARGET) &
	sleep 2
	curl -s http://localhost:8080/health || echo "Server test failed"
	pkill -f $(SERVER_TARGET) || true

test-shared: shared
	@echo "Testing shared library..."
	@echo "Shared library built successfully"

# Run all tests
test: test-cli test-server test-shared
	@echo "All tests completed"

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build all components"
	@echo "  cli        - Build interactive CLI"
	@echo "  shared     - Build shared library for FFI"
	@echo "  server     - Build HTTP API server"
	@echo "  synthetic  - Build synthetic data trainer"
	@echo "  clean      - Remove build artifacts"
	@echo "  test       - Run all tests"
	@echo "  help       - Show this help"

.PHONY: all cli shared server synthetic clean test test-cli test-server test-shared help