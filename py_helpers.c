// C helper functions to bridge Fortran and Python using direct C API calls
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

// Global references to keep Python objects alive between calls if needed
// We might not need the function object globally if we fetch it each time,
// but keeping module reference can be slightly more efficient.
static PyObject *global_module_obj = NULL;
static PyObject *global_inference_func_obj = NULL;

// Helper function to import NumPy API. Must be called once after Py_Initialize
int import_numpy_api() {
    // PyArray_API is a macro that expands to initializing a static variable.
    // import_array() needs to be called in the function that uses NumPy C API.
    // It's generally recommended to call it once per C extension module initialization.
    // Here, we'll call it within py_initialize for simplicity.
    import_array();
    if (PyErr_Occurred()) {
        fprintf(stderr, "C Helper Error: Failed to import NumPy C-API.\n");
        PyErr_Print();
        return -1;
    }
    return 0;
}

void py_initialize() {
    if (!Py_IsInitialized()) {
        Py_InitializeEx(0); // Initialize Python (0 = dont init signals)


        if (import_numpy_api() < 0) {
            Py_FinalizeEx(); // Clean up Python if NumPy fails
            fprintf(stderr, "C Helper FATAL: NumPy C-API init failed. Exiting.\n");
            exit(1); 
        }

        // Add current directory to Python path to find test_NN.py
        PyObject *sys_path = PySys_GetObject("path");
        if (sys_path == NULL) {
            fprintf(stderr, "C Helper Error: Failed to get sys.path\n");
            PyErr_Print();
        } else {
            PyObject *path_obj = PyUnicode_FromString("."); // Current directory
            if (path_obj == NULL) {
                fprintf(stderr, "C Helper Error: Failed create path string\n");
                PyErr_Print();
            } else {
                if (PyList_Insert(sys_path, 0, path_obj) < 0) { 
                    fprintf(stderr, "C Helper Error: Failed to add . to sys.path\n");
                    PyErr_Print();
                }
                Py_DECREF(path_obj); 
            }
        }
        printf("C Helper: Python and NumPy C-API Initialized, path set.\n");

        // Pre-load module and function object
        global_module_obj = PyImport_ImportModule("test_NN");
        if (global_module_obj == NULL) {
             fprintf(stderr, "C Helper Error: Failed to import module 'test_NN' during init.\n");
             PyErr_Print(); 
        } else {
            global_inference_func_obj = PyObject_GetAttrString(global_module_obj, "run_resnet_inference");
            if (global_inference_func_obj == NULL || !PyCallable_Check(global_inference_func_obj)) {
                fprintf(stderr, "C Helper Error: Cannot find callable function 'run_resnet_inference' during init.\n");
                PyErr_Print();
                Py_XDECREF(global_inference_func_obj);
                global_inference_func_obj = NULL; // Ensure it's NULL if failed
            }
        }

    } else {
        printf("C Helper: Python already initialized.\n");
    }
}

// --- Finalization --- 
void py_finalize() {
    if (Py_IsInitialized()) {
        // Decrement references to stored Python objects before finalizing
        Py_XDECREF(global_inference_func_obj);
        Py_XDECREF(global_module_obj);
        global_inference_func_obj = NULL;
        global_module_obj = NULL;

        Py_FinalizeEx();
        printf("C Helper: Python Finalized.\n");
    } else {
        printf("C Helper: Python not initialized, nothing to finalize.\n");
    }
}

// This function is called directly from Fortran
void execute_inference(float* input_ptr, size_t input_size, float* output_ptr, size_t output_size) {
    
    PyObject *p_input_numpy = NULL;
    PyObject *p_input_tensor = NULL;
    PyObject *p_torch_module = NULL;
    PyObject *p_from_numpy_func = NULL;
    PyObject *p_args = NULL;
    PyObject *p_output_tensor = NULL;
    PyObject *p_output_numpy = NULL;
    PyGILState_STATE gstate; // Variable to hold GIL state

    // --- Pre-check --- 
    if (!Py_IsInitialized()) {
        fprintf(stderr, "C execute_inference Error: Python not initialized!\n");
        return;
    }
    if (global_inference_func_obj == NULL) {
         fprintf(stderr, "C execute_inference Error: Python function not loaded!\n");
        return;
    }

    // Expected shapes and sizes (hardcoded for this example)
    npy_intp input_dims[] = {1, 3, 224, 224}; // NumPy dimensions type
    size_t expected_input_size = 1 * 3 * 224 * 224;
    size_t expected_output_size = 1 * 1000;

    if (input_size != expected_input_size) {
        fprintf(stderr, "C execute_inference Error: Incorrect input size. Expected %zu, Got %zu\n", expected_input_size, input_size);
        return;
    }
     if (output_size != expected_output_size) {
        fprintf(stderr, "C execute_inference Error: Incorrect output buffer size. Expected %zu, Got %zu\n", expected_output_size, output_size);
        return;
    }

    // --- Acquire GIL --- 
    // All Python C API calls must be between Ensure/Release
    gstate = PyGILState_Ensure();

    // --- Input: C pointer -> NumPy array -> Torch Tensor --- 
    
    // 1. Wrap C input pointer in a NumPy array (no copy)
    // NPY_FLOAT corresponds to C float
    p_input_numpy = PyArray_SimpleNewFromData(4, input_dims, NPY_FLOAT, (void*)input_ptr);
    if (p_input_numpy == NULL) {
        fprintf(stderr, "C execute_inference Error: Failed to create NumPy array from input data.\n");
        PyErr_Print();
        goto cleanup;
    }

    // 2. Convert NumPy array to PyTorch Tensor (using torch.from_numpy)
    p_torch_module = PyImport_ImportModule("torch"); 
    if (p_torch_module == NULL) {
        fprintf(stderr, "C execute_inference Error: Failed to import torch module.\n");
        PyErr_Print();
        goto cleanup;
    }
    p_from_numpy_func = PyObject_GetAttrString(p_torch_module, "from_numpy");
    if (p_from_numpy_func == NULL || !PyCallable_Check(p_from_numpy_func)) {
        fprintf(stderr, "C execute_inference Error: Failed to get torch.from_numpy function.\n");
        PyErr_Print();
        goto cleanup;
    }
    p_input_tensor = PyObject_CallFunctionObjArgs(p_from_numpy_func, p_input_numpy, NULL);
    if (p_input_tensor == NULL) {
        fprintf(stderr, "C execute_inference Error: Call to torch.from_numpy failed.\n");
        PyErr_Print();
        goto cleanup;
    }

    // --- Call Python Inference Function --- 
    p_args = Py_BuildValue("(O)", p_input_tensor);
    if (p_args == NULL) {
        fprintf(stderr, "C execute_inference Error: Failed to build arguments tuple.\n");
        PyErr_Print();
        goto cleanup;
    }
    
    printf("C execute_inference: Calling Python run_resnet_inference...\n");
    p_output_tensor = PyObject_CallObject(global_inference_func_obj, p_args);
    if (p_output_tensor == NULL) {
        fprintf(stderr, "C execute_inference Error: Call to run_resnet_inference failed.\n");
        PyErr_Print();
        goto cleanup;
    }
    printf("C execute_inference: Returned from Python function.\n");

    // --- Output: Torch Tensor -> NumPy array -> C pointer --- 

    // 1. Convert output tensor to NumPy array (using tensor.numpy() method)
    p_output_numpy = PyObject_CallMethod(p_output_tensor, "numpy", NULL); 
    if (p_output_numpy == NULL) {
        fprintf(stderr, "C execute_inference Error: Call to tensor.numpy() failed.\n");
        PyErr_Print();
        goto cleanup;
    }
    // Check if it's actually a NumPy array
    if (!PyArray_Check(p_output_numpy)) {
         fprintf(stderr, "C execute_inference Error: tensor.numpy() did not return a NumPy array.\n");
         goto cleanup;
    }
    // Check if the data type is float32 (NPY_FLOAT)
    if (PyArray_TYPE((PyArrayObject*)p_output_numpy) != NPY_FLOAT) {
        fprintf(stderr, "C execute_inference Error: Output NumPy array is not float32 type.\n");
        goto cleanup;
    }
    // Check if the array is C-contiguous (needed for direct memory copy)
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)p_output_numpy)) {
         fprintf(stderr, "C execute_inference Error: Output NumPy array is not C-contiguous.\n");
        goto cleanup;
    }

    // 2. Copy data from NumPy array to the C output pointer
    float* output_numpy_data = (float*)PyArray_DATA((PyArrayObject*)p_output_numpy);
    if (output_numpy_data == NULL) {
        fprintf(stderr, "C execute_inference Error: Failed to get data pointer from output NumPy array.\n");
        goto cleanup;
    }
    // Check size matches again before copying
     if ((size_t)PyArray_SIZE((PyArrayObject*)p_output_numpy) != expected_output_size) {
        fprintf(stderr, "C execute_inference Error: Output NumPy array size (%zd) doesn't match expected (%zu).\n", 
                PyArray_SIZE((PyArrayObject*)p_output_numpy), expected_output_size);
        goto cleanup;
    }

    memcpy(output_ptr, output_numpy_data, expected_output_size * sizeof(float));
    printf("C execute_inference: Copied output data back to Fortran pointer.\n");


// --- Cleanup --- 
cleanup:
    // Decrement references for objects created in this function
    Py_XDECREF(p_output_numpy);
    Py_XDECREF(p_output_tensor);
    Py_XDECREF(p_args);
    Py_XDECREF(p_input_tensor);
    Py_XDECREF(p_from_numpy_func);
    Py_XDECREF(p_torch_module);
    Py_XDECREF(p_input_numpy);
    
    // --- Release GIL --- 
    PyGILState_Release(gstate);
} 
