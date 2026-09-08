function(minirt_add_test test_name source)
    set(target_name "minirt_${test_name}")

    add_executable("${target_name}" "${source}")
    target_link_libraries("${target_name}" PRIVATE minirt::runtime)
    target_include_directories("${target_name}" PRIVATE "${PROJECT_SOURCE_DIR}/tests/support")

    add_test(
        NAME "${test_name}"
        COMMAND "$<TARGET_FILE:${target_name}>"
    )
    set_tests_properties("${test_name}" PROPERTIES TIMEOUT 30)
endfunction()

function(minirt_add_internal_test test_name source)
    minirt_add_test("${test_name}" "${source}")
    target_include_directories(
        "minirt_${test_name}"
        PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )
endfunction()

minirt_add_internal_test(runtime_wrappers runtime/runtime_wrappers_test.cpp)
minirt_add_internal_test(device_buffer memory/device_buffer_test.cu)

set_tests_properties(
    runtime_wrappers
    device_buffer
    PROPERTIES LABELS "cuda"
)
