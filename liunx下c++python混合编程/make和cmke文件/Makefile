#python version
PYTHON_VERSION = 3.5
PYTHON_VERSIONT = 35
# location of the Python header files
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
# location of the Boost Python include files and library
BOOST_INC = /usr/local/include/boost
BOOST_LIB = /usr/lib
# compile mesh classes
#project save path
PROPATH = ~/hello
#project name
TARGET = ImageProcessAPI
#.o to .so
$(TARGET).so: $(TARGET).o
	g++ -shared -Wl,--export-dynamic $(TARGET).o `pkg-config --cflags --libs opencv` -L$(BOOST_LIB) -lboost_python$(PYTHON_VERSIONT) -L/usr/lib/python$(PYTHON_VERSION)/config-3.5m-x86_64-linux-gnu -lpython$(PYTHON_VERSION) -o $(TARGET).so

#file to .o(include all .h file and .cpp file)
$(TARGET).o:OpencvConversion.h $(TARGET).cpp 
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).cpp

#clear .o file
clean : 
	rm -rf $(PROPATH)/$(TARGET).o
