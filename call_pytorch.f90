program call_pytorch_from_fortran
  use iso_c_binding
  implicit none

  ! --- Interfaces to C helper functions (MUST BE IMPLEMENTED IN C) ---

  interface
     ! Initializes the Python interpreter and NumPy API
     subroutine py_initialize() bind(C, name='py_initialize')
     end subroutine py_initialize

     ! Finalizes the Python interpreter
     subroutine py_finalize() bind(C, name='py_finalize')
     end subroutine py_finalize

     ! Executes the inference directly via C API calls
     subroutine execute_inference(input_ptr, input_size, output_ptr, output_size) &
          bind(C, name='execute_inference')
        import :: c_float, c_size_t, c_ptr
        type(c_ptr), value             :: input_ptr
        integer(c_size_t), value      :: input_size
        type(c_ptr), value             :: output_ptr
        integer(c_size_t), value      :: output_size
     end subroutine execute_inference
  end interface

  ! --- Fortran Variables ---
  real(c_float), allocatable, target :: input_data(:)
  real(c_float), allocatable, target :: output_data(:)

  integer(c_size_t) :: input_size
  integer(c_size_t) :: output_size

  ! --- Define Array Sizes ---
  ! Input: (1, 3, 224, 224) -> Flattened size
  input_size = 1 * 3 * 224 * 224
  ! Output: (1, 1000) -> Flattened size
  output_size = 1 * 1000

  print *, "Fortran: Starting..."
  print *, "Fortran: Input array size:", input_size
  print *, "Fortran: Output array size:", output_size

  ! --- Allocate and Initialize Arrays ---
  allocate(input_data(input_size))
  allocate(output_data(output_size))

  ! Initialize input data (e.g., with ones, matching Python example)
  input_data = 1.0_c_float
  print *, "Fortran: Input data initialized with ones."

  ! Initialize output data (optional, e.g., to NaN to see changes)
  output_data = -999.0_c_float ! Or some other indicator

  ! --- Initialize Python ---
  print *, "Fortran: Initializing Python interpreter & NumPy API..."
  call py_initialize() ! Call C helper
  print *, "Fortran: Python interpreter & NumPy API initialized."

  ! --- Call the Python Function via Direct C Interface ---
  print *, "Fortran: Calling C function execute_inference..."
  call execute_inference(c_loc(input_data), input_size, &
                         c_loc(output_data), output_size)
  print *, "Fortran: Returned from C function."

  ! --- Print Some Results ---
  print *, "Fortran: Output data (first 10 elements):"
  print '(10(F8.4, X))', output_data(1:10)
  print *, "Fortran: Output data (last 10 elements):"
  print '(10(F8.4, X))', output_data(output_size-9:output_size)

  ! --- Clean Up ---
  print *, "Fortran: Finalizing Python interpreter..."
  call py_finalize() ! Call C helper
  print *, "Fortran: Python interpreter finalized."

  deallocate(input_data)
  deallocate(output_data)

  print *, "Fortran: Finished."

end program call_pytorch_from_fortran 