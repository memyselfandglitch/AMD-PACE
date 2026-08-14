add_executable(pace_pv_brgemm_dataflow pv_brgemm_dataflow.cpp)
add_dependencies(pace_pv_brgemm_dataflow jit_backend)
target_compile_features(pace_pv_brgemm_dataflow PRIVATE cxx_std_17)
target_compile_options(
  pace_pv_brgemm_dataflow
  PRIVATE -O3 -mavx2 -mavx512f -mavx512bw -mavx512dq -mavx512vl
          -mavx512bf16)
target_include_directories(
  pace_pv_brgemm_dataflow PRIVATE ${JIT_BACKEND_INCLUDE_DIR})
target_link_libraries(
  pace_pv_brgemm_dataflow
  PRIVATE ${JIT_BACKEND_STATIC_LIB} OpenMP::OpenMP_CXX Threads::Threads
          ${CMAKE_DL_LIBS})
