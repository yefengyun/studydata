# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(ImageProcessAPI)

# Find OpenC
find_package(OpenCV REQUIRED)

# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


set(BOOST_INCLUDEDIR /usr/local/include/boost)
set(BOOST_LIBRARYDIR /usr/local/lib)
find_package(Boost 1.68.0 COMPONENTS python35 REQUIRED) 
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

message(STATUS "Boost library status:")
message(STATUS "    Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
message(STATUS "    Boost_LIBRARIES = ${Boost_LIBRARIES}.")
message(STATUS "    Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")

set(PYTHON_INCLUDEDIR /usr/include/python)
#set(PYTHON_LIBRARYDIR /usr/lib/python3.5/config-3.5m-x86_64-linux-gnu)
set(PYTHON_ROOT /usr)
find_package(PythonLibs REQUIRED) 
include_directories(${PYTHON_INCLUDE_DIRS})

add_library(ImageProcessAPI SHARED ImageProcessAPI.cpp OpencvConversion.h)
# Declare the executable target built from your sources
#add_executable(ImageProcessAPI ImageProcessAPI.cpp OpencvConversion.h)

# Link your application with OpenCV libraries
#set(SOURCE_FILES greet_ext.cpp greet.cpp greet.h)
target_link_libraries(ImageProcessAPI ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBRARIES})


#set(CMAKE_CXX_FLAGS_RELEASE "-Wall -std=c++11")
#set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -std=c++11")
set(CMAKE_BUILD_TYPE Debug)
