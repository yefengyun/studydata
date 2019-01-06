// ImageProcess.cpp : 定义 DLL 应用程序的导出函数。
#include "OpencvConversion.h"
#include <Python.h>
#include <boost/python.hpp>

using namespace std;
using namespace boost;
using namespace boost::python;

class ImageProcess_Python 
{
//类变量区
public:
	std::string img_path;

//类函数区
public:
	ImageProcess_Python() {}
	~ImageProcess_Python() {}

	string test()
	{
		return "Load Sucessful";
	}
	
	PyObject * ImageCorrect_Python(PyObject * pysrc)
	{
		NDArrayConverter ndac;
		cv::Mat src = ndac.toMat(pysrc);
		return ndac.toNDArray(src);
	}
};

BOOST_PYTHON_MODULE(libImageProcessAPI)
{
	class_<ImageProcess_Python>("ImageProcess")//默认无参构造函数
		// .def(init<std::string>())//init构造函数
		.def("test", &ImageProcess_Python::test)//不带参数的方法
		.def("ImageCorrect", &ImageProcess_Python::ImageCorrect_Python)
		.def_readwrite("img_path", &ImageProcess_Python::img_path);//可读写成员变量
}