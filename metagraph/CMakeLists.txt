cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0015 NEW)

# Hardcoded version
set(VERSION "0.3.6")
message(STATUS "Current MetaGraph version: ${VERSION}")

project(MetaGraph
  VERSION ${VERSION}
  DESCRIPTION "A tool for scalable construction of annotated genome graphs and sequence-to-graph alignment"
)

option(BUILD_STATIC "Build a static binary" OFF)

set(SRC_DIR "src")
message(STATUS "Building using source directory: ${SRC_DIR}")

if(BUILD_STATIC)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
  set(OPENSSL_USE_STATIC_LIBS TRUE)
  set(Boost_USE_STATIC_LIBS ON)
endif()

add_definitions(-DVERSION="${VERSION}")
include(CheckFunctionExists)
include(CheckLibraryExists)
include(CheckCXXSourceRuns) # Required for filesystem check
include(CMakeListsHelpers.txt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enableCCache()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Default alphabet (keep for compatibility)
if(NOT CMAKE_DBG_ALPHABET)
  set(CMAKE_DBG_ALPHABET "DNA")
endif()

if(CMAKE_DBG_ALPHABET STREQUAL "DNA")
  add_definitions(-D_DNA_GRAPH)
elseif(CMAKE_DBG_ALPHABET STREQUAL "DNA5")
  add_definitions(-D_DNA5_GRAPH)
elseif(CMAKE_DBG_ALPHABET STREQUAL "DNA_CASE_SENSITIVE")
  add_definitions(-D_DNA_CASE_SENSITIVE_GRAPH)
else()
  add_definitions(-D_PROTEIN_GRAPH)
endif()

# JsonCpp
set(JSONCPP_WITH_TESTS FALSE)
add_subdirectory(external-libraries/jsoncpp)
# Force JsonCpp to build with C++17 to support string_view
if(TARGET jsoncpp_lib)
  set_target_properties(jsoncpp_lib PROPERTIES CXX_STANDARD 17)
endif()
if(TARGET jsoncpp_static)
  set_target_properties(jsoncpp_static PROPERTIES CXX_STANDARD 17)
endif()
if(TARGET jsoncpp_object)
  set_target_properties(jsoncpp_object PROPERTIES CXX_STANDARD 17)
endif()

add_compile_options(
  -Wall -Wextra
  -D_THREAD_SAFE -pthread
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wno-exit-time-destructors
    -Wno-deprecated-declarations
  )
  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    add_compile_options(
     -Wno-suggest-destructor-override
     -Wno-deprecated-copy
     -Wno-deprecated-declarations
    )
  endif()
endif()

if(APPLE)
  include_directories(SYSTEM /usr/local/include)
  include_directories(SYSTEM /opt/homebrew/include)
  link_directories(/usr/local/lib)
  if(EXISTS /opt/homebrew/lib)
    link_directories(/opt/homebrew/lib)
  endif()
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-Wno-poison-system-directories SUPPORT_NO_POISON_SYS_DIR)
  if(SUPPORT_NO_POISON_SYS_DIR)
    add_compile_options(-Wno-poison-system-directories)
  endif()
endif()

findOpenMP()
if("${OpenMP_CXX_VERSION}" VERSION_GREATER_EQUAL 5.0)
  add_definitions(-D_OpenMP_5)
endif()

# Check for 128-bit integers
try_compile(MODE_TI
  "${PROJECT_BINARY_DIR}"
  "${PROJECT_SOURCE_DIR}/external-libraries/sdsl-lite/CMakeModules/check_mode_ti.cpp"
)
if(MODE_TI)
  add_definitions(-DMODE_TI)
  if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
      add_compile_options(-mcx16)
  endif()
endif()

# CPU features
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(STRINGS "/proc/cpuinfo" _cpuinfo)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  execute_process(COMMAND sysctl -n machdep.cpu.features
                  OUTPUT_VARIABLE _cpuinfo OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND sysctl -n machdep.cpu.leaf7_features
                  OUTPUT_VARIABLE _cpuinfo2 OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(_cpuinfo "${_cpuinfo} ${_cpuinfo2}")
endif()

if(WITH_MSSE42 OR (NOT DEFINED WITH_MSSE42 AND _cpuinfo MATCHES "(sse4_2)|(sse4a)|(SSE4.2)"))
  add_compile_options(-msse4.2)
endif()

if(WITH_AVX OR (NOT DEFINED WITH_AVX AND _cpuinfo MATCHES "(avx2)|(AVX2)"))
  add_compile_options(-mavx -mavx2 -mfma -mbmi -mbmi2)
endif()

# KMC k-mer counter


link_directories(
  external-libraries/sdsl-lite/lib
)

include_directories(
 PUBLIC
  external-libraries/sdsl-lite/include
  external-libraries/hopscotch-map/include
  external-libraries/cpp_progress_bar
  external-libraries/spdlog/include
  external-libraries/zlib
  external-libraries/jsoncpp/include
  ${PROJECT_SOURCE_DIR}/${SRC_DIR}
)

include_directories(
  PUBLIC SYSTEM
  external-libraries/asio/asio/include
  external-libraries/simde-no-tests
  external-libraries/asio/asio/include
  external-libraries/simde-no-tests
)

add_library(ips4o INTERFACE)
target_include_directories(ips4o INTERFACE external-libraries/ips4o)



add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE external-libraries/eigen)
target_compile_options(eigen INTERFACE -Wno-unused-but-set-variable -Wno-unused-parameter)

# ASIO
add_definitions(-DASIO_STANDALONE -DUSE_STANDALONE_ASIO)
set(USE_STANDALONE_ASIO ON)
set(ASIO_PATH external-libraries/asio/asio)

# Pthreads
if(APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

add_subdirectory(external-libraries/spdlog)

add_subdirectory(external-libraries/DYNAMIC)
add_subdirectory(external-libraries/zlib)
add_subdirectory(external-libraries/sdsl-lite)

target_compile_options(zlib
  PRIVATE
    -Wno-unused-variable
    -Wno-sign-compare
    -Wno-implicit-fallthrough
)

add_library(progress_bar STATIC
  external-libraries/cpp_progress_bar/progress_bar.cpp
)



# Build HTSLIB using custom target (simplification)
set(HTSLIB_CONFIGURE_FLAGS --disable-libcurl --disable-bz2 --disable-lzma)
if(BUILD_STATIC)
  list(APPEND HTSLIB_CONFIGURE_FLAGS --without-openssl)
endif()

add_custom_target(HTSLIB_BUILD
  COMMAND autoreconf -i
  COMMAND ./configure ${HTSLIB_CONFIGURE_FLAGS}
  COMMAND ${CMAKE_MAKE_PROGRAM}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external-libraries/htslib
  COMMENT "Building HTSLib"
)

add_library(HTSLIB STATIC IMPORTED GLOBAL)
set_target_properties(HTSLIB PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/external-libraries/htslib/libhts.a
)
add_dependencies(HTSLIB HTSLIB_BUILD)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external-libraries/htslib)

# Gather source files
file(GLOB_RECURSE metagraph_core_files
  "${SRC_DIR}/*.cpp"
)

# Exclude main and unused
list(FILTER metagraph_core_files EXCLUDE REGEX ".*\\._.*")
list(FILTER metagraph_core_files EXCLUDE REGEX ".*main.cpp")

# Split CLI files
set(metagraph_cli_files ${metagraph_core_files})
list(FILTER metagraph_cli_files INCLUDE REGEX "${SRC_DIR}/cli/.*")
list(FILTER metagraph_core_files EXCLUDE REGEX "${SRC_DIR}/cli/.*")

add_library(metagraph-core STATIC ${metagraph_core_files})
add_library(metagraph-cli STATIC ${metagraph_cli_files})

add_dependencies(metagraph-core HTSLIB)

add_executable(metagraph "${SRC_DIR}/main.cpp")
# To rename the binary, uncomment and edit the following line:
# set_target_properties(metagraph PROPERTIES OUTPUT_NAME "your_name_here")

if(BUILD_STATIC)
  set(JSONCPP_LIB jsoncpp_static)
else()
  set(JSONCPP_LIB jsoncpp_lib)
endif()

set(METALIBS
  HTSLIB zlib deflate
  sdsl
  ${JSONCPP_LIB}
  progress_bar
  spdlog::spdlog
  XXSDS_DYNAMIC
  ips4o
  eigen
)

if(APPLE)
  list(APPEND METALIBS ${OpenMP_CXX_LIBRARIES})
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  list(APPEND METALIBS -latomic)
endif()

# Static build on Linux (simplified)
# Assuming dynamic link is fine unless specified, keeping original logic for consistency
if(BUILD_STATIC AND NOT APPLE)
   set(USE_OPENSSL OFF CACHE BOOL "Disable SSL in Simple-Web-Server" FORCE)
   
   # Use absolute paths for static libs
   list(APPEND METALIBS "/usr/lib/x86_64-linux-gnu/libssl.a" "/usr/lib/x86_64-linux-gnu/libcrypto.a" "/usr/lib/gcc/x86_64-linux-gnu/9/libgomp.a" -lrt -ldl)
   
   target_link_options(metagraph PRIVATE "-static" "-static-libstdc++" "-static-libgcc")
   # Ensure all symbols are bundled
   list(APPEND METALIBS -Wl,--whole-archive -lpthread -Wl,--no-whole-archive)
endif()

# Checking for filesystem
set(CMAKE_REQUIRED_FLAGS " -std=c++17")
set(CMAKE_REQUIRED_LIBRARIES "stdc++fs")
check_cxx_source_runs("
    #include <iostream>
    #include <filesystem>
    int main() {
        std::cout << std::filesystem::temp_directory_path();
        return 0;
    }
" STDCPPFS)
if(STDCPPFS)
    list(APPEND METALIBS -lstdc++fs)
endif()

target_link_libraries(metagraph-core ${METALIBS})

# Simple Web Server
set(USE_OPENSSL OFF CACHE BOOL "Disable SSL in Simple-Web-Server" FORCE)
add_subdirectory(external-libraries/Simple-Web-Server)
target_link_libraries(metagraph-cli PUBLIC metagraph-core simple-web-server)

target_link_libraries(metagraph metagraph-core metagraph-cli)



# Install
file(GLOB_RECURSE header_files "${SRC_DIR}/*.hpp")
install(TARGETS metagraph-core DESTINATION lib)
install(FILES ${header_files} DESTINATION include/metagraph)
install(TARGETS metagraph DESTINATION bin)
