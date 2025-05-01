FC = gfortran
CC = gcc

# --- Python Configuration ---
PYTHON_CFLAGS := $(shell python3-config --cflags)
PYTHON_LDFLAGS := $(shell python3-config --ldflags --embed)

ifeq ($(PYTHON_CFLAGS),)
    $(error "python3-config --cflags failed. Is python3-config in PATH and working?")
endif
ifeq ($(PYTHON_LDFLAGS),)
    $(error "python3-config --ldflags --embed failed. Is python3-config in PATH and working?")
endif

# --- NumPy Configuration ---
NUMPY_INCLUDE := $(shell python3 -c "import numpy; print(numpy.get_include())" 2>/dev/null)
ifeq ($(NUMPY_INCLUDE),)
    $(error "Failed to get NumPy include path. Is NumPy installed in the python3 environment?")
endif
NUMPY_CFLAGS = -I"$(NUMPY_INCLUDE)"

# --- Flags ---
# Fortran flags (add optimization, standards flags etc. as needed)
FFLAGS = -Wall -Wextra -fcheck=all -fPIC
# C flags (include Python and NumPy headers, enable PIC)
CFLAGS = -Wall -Wextra -fPIC $(PYTHON_CFLAGS) $(NUMPY_CFLAGS)
# Linker flags (link against Python library for embedding)
LDFLAGS = $(PYTHON_LDFLAGS)

# --- Files ---
# Executable name
EXEC = fortran_pytorch_test

# Source files (in the current directory)
F_SOURCES = call_pytorch.f90
C_SOURCES = py_helpers.c

# Object files (derived from sources)
F_OBJECTS = $(F_SOURCES:.f90=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)
OBJECTS = $(F_OBJECTS) $(C_OBJECTS)

# --- Rules ---

# Default target: build the executable
.PHONY: all
all: $(EXEC)

# Rule to link the executable
$(EXEC): $(OBJECTS)
	@echo "Linking $@..."
	$(FC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Build complete: $(EXEC)"

# Rule to compile Fortran source files
%.o: %.f90
	@echo "Compiling $< -> $@"
	$(FC) -c $< -o $@ $(FFLAGS)

# Rule to compile C source files
%.o: %.c
	@echo "Compiling $< -> $@"
	$(CC) -c $< -o $@ $(CFLAGS)

# Target to run the executable
.PHONY: run
run: $(EXEC)
	@echo "Running $(EXEC)..."
	@echo "-----------------------------------------------------"
	# Set PYTHONPATH to include the current directory containing test_NN.py
	PYTHONPATH=.:$(PYTHONPATH) ./$(EXEC)
	@echo "-----------------------------------------------------"

# Target to clean up generated files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -f $(EXEC) $(OBJECTS) *.mod *~

# Target for clean build and run
.PHONY: new
new: clean all run
