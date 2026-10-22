CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra -Wpedantic
LDFLAGS ?=

TARGET := log_reg
SOURCES := main.cpp matrix.cpp logistic_regression.cpp
OBJECTS := $(SOURCES:.cpp=.o)
TEST_TARGET := log_reg_tests
TEST_SOURCES := tests.cpp matrix.cpp logistic_regression.cpp

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

%.o: %.cpp matrix.h logistic_regression.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

$(TEST_TARGET): $(TEST_SOURCES)
	$(CXX) $(CXXFLAGS) $(TEST_SOURCES) $(LDFLAGS) -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	$(RM) $(OBJECTS) $(TARGET) $(TEST_TARGET)
