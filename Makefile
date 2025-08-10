# Makefile for Micro Transformer with OpenMP (C++23)

# Compiler settings
CXX = "C:/Users/$(USERNAME)/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/g++.exe"
CXXFLAGS = -std=c++23 -fopenmp -Wall -Wextra -O3
CXXFLAGS_DEBUG = -std=c++23 -fopenmp -Wall -Wextra -g -DDEBUG

# Directory settings
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Target programs
TARGET = micro_transformer
TARGET_DEBUG = micro_transformer_debug

# Default target
all: $(TARGET)

# Release version
$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Debug version
debug: CXXFLAGS = $(CXXFLAGS_DEBUG)
debug: $(TARGET_DEBUG)

$(TARGET_DEBUG): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_DEBUG) -o $@ $^

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	mkdir $(BUILD_DIR)

# Test C++23 support
test-cpp23:
	$(CXX) -std=c++23 -dM -E -x c++ - < NUL | findstr "__cplusplus"

# Test OpenMP support
test-openmp:
	$(CXX) -std=c++23 -fopenmp -o test_openmp_temp -x c++ - -DTEST_OPENMP && ./test_openmp_temp && del test_openmp_temp.exe
	@echo "int main(){return 0;}" | $(CXX) -std=c++23 -fopenmp -o test_openmp_temp -x c++ - && echo "OpenMP support OK" && del test_openmp_temp.exe

# Clean up
clean:
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
	if exist $(TARGET).exe del $(TARGET).exe
	if exist $(TARGET_DEBUG).exe del $(TARGET_DEBUG).exe
	if exist *.exe del *.exe

# Help information
help:
	@echo "Available targets:"
	@echo "  all          - Build release version"
	@echo "  debug        - Build debug version"
	@echo "  test-cpp23   - Test C++23 support"
	@echo "  test-openmp  - Test OpenMP support"
	@echo "  clean        - Clean build files"
	@echo "  help         - Show this help information"

.PHONY: all debug clean test-cpp23 test-openmp help
