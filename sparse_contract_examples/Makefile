# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Iinclude -std=c++11 -O3 -g

# Source files
SRCS = $(wildcard *.cpp lib/*.cpp)

# Object files directory
BUILD_DIR = build

# Object files
OBJS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Target executable
TARGET = sc

# Default rule
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# Compiling
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf $(BUILD_DIR)
