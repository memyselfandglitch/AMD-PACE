function(add_pace_brgemm_benchmark target source)
  add_executable(${target} ${source})
  add_dependencies(${target} jit_backend)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(
    ${target}
    PRIVATE -O3 -mavx2 -mavx512f -mavx512bw -mavx512dq -mavx512vl
            -mavx512bf16)
  target_include_directories(${target} PRIVATE ${JIT_BACKEND_INCLUDE_DIR})
  target_link_libraries(
    ${target}
    PRIVATE ${JIT_BACKEND_STATIC_LIB} OpenMP::OpenMP_CXX Threads::Threads
            ${CMAKE_DL_LIBS})
endfunction()

function(add_pace_avx512_benchmark target source)
  add_executable(${target} ${source})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(
    ${target}
    PRIVATE -O3 -mavx2 -mavx512f -mavx512bw -mavx512dq -mavx512vl
            -mavx512bf16)
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
  target_link_libraries(
    ${target} PRIVATE OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
endfunction()

add_pace_brgemm_benchmark(pace_pv_brgemm_dataflow pv_brgemm_dataflow.cpp)
add_pace_brgemm_benchmark(pace_qk_brgemm_dataflow qk_brgemm_dataflow.cpp)
add_pace_avx512_benchmark(
  pace_decode_layout_traversal decode_layout_traversal.cpp)
