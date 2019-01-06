//数据转换
# ifndef __OPENCVCOVERSION_H__
# define __OPENCVCOVERSION_H__

#pragma once

#include <Python.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

using namespace cv;

#if PY_MAJOR_VERSION >= 3
// Python3 treats all ints as longs, PyInt_X functions have been removed.
#define PyInt_Check PyLong_Check
#define PyInt_CheckExact PyLong_CheckExact
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyInt_FromLong PyLong_FromLong
#define PyNumber_Int PyNumber_Long

// Python3 strings are unicode, these defines mimic the Python2 functionality.
#define PyString_Check PyUnicode_Check
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyString_Size PyUnicode_GET_SIZE

// PyUnicode_AsUTF8 isn't available until Python 3.3
#if (PY_VERSION_HEX < 0x03030000)
#define PyString_AsString _PyUnicode_AsString
#else
#define PyString_AsString PyUnicode_AsUTF8
#endif
#endif

#define NUMPY_IMPORT_ARRAY_RETVAL

//===================    MACROS    =================================================================
#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

//===================   ERROR HANDLING     =========================================================
static PyObject* opencv_error = 0;
static int failmsg(const char *fmt, ...)
{
	char str[1000];

	va_list ap;
	va_start(ap, fmt);
	vsnprintf(str, sizeof(str), fmt, ap);
	va_end(ap);

	PyErr_SetString(PyExc_TypeError, str);
	return 0;
}
// static PyObject* failmsgp(const char *fmt, ...)
// {
// 	char str[1000];

// 	va_list ap;
// 	va_start(ap, fmt);
// 	vsnprintf(str, sizeof(str), fmt, ap);
// 	va_end(ap);

// 	PyErr_SetString(PyExc_TypeError, str);
// 	return 0;
// }

//===================   THREADING     ==============================================================
class PyAllowThreads
{
public:
	PyAllowThreads() : _state(PyEval_SaveThread()) {}
	~PyAllowThreads()
	{
		PyEval_RestoreThread(_state);
	}
private:
	PyThreadState* _state;
};
class PyEnsureGIL
{
public:
	PyEnsureGIL() : _state(PyGILState_Ensure()) {}
	~PyEnsureGIL()
	{
		PyGILState_Release(_state);
	}
private:
	PyGILState_STATE _state;
};

//===================   REFCOUNT(引用计数)  ==============================================================
static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
(0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0") * sizeof(int);
static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
	return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}
static inline int* refcountFromPyObject(const PyObject* obj)
{
	return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

//===================   NUMPY ALLOCATOR FOR OPENCV     =============================================
class NumpyAllocator : public cv::MatAllocator
{
public:
	NumpyAllocator() {
		stdAllocator = cv::Mat::getStdAllocator();
	}
	~NumpyAllocator() {
	}

	UMatData* allocate(PyObject* o, int dims, const int* sizes, int type,
		size_t* step) const {
		UMatData* u = new UMatData(this);
		u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*)o);
		npy_intp* _strides = PyArray_STRIDES((PyArrayObject*)o);
		for (int i = 0; i < dims - 1; i++)
			step[i] = (size_t)_strides[i];
		step[dims - 1] = CV_ELEM_SIZE(type);
		u->size = sizes[0] * step[0];
		u->userdata = o;
		return u;
	}

	UMatData* allocate(int dims0, const int* sizes, int type, void* data,
		size_t* step, int flags, cv::UMatUsageFlags usageFlags) const {
		if (data != 0) {
			CV_Error(Error::StsAssert, "The data should normally be NULL!");
			// probably this is safe to do in such extreme case
			return stdAllocator->allocate(dims0, sizes, type, data, step, flags,
				usageFlags);
		}
		PyEnsureGIL gil;

		int depth = CV_MAT_DEPTH(type);
		int cn = CV_MAT_CN(type);
		const int f = (int)(sizeof(size_t) / 8);
		int typenum =
			depth == CV_8U ? NPY_UBYTE :
			depth == CV_8S ? NPY_BYTE :
			depth == CV_16U ? NPY_USHORT :
			depth == CV_16S ? NPY_SHORT :
			depth == CV_32S ? NPY_INT :
			depth == CV_32F ? NPY_FLOAT :
			depth == CV_64F ?
			NPY_DOUBLE :
			f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
		int i, dims = dims0;
		cv::AutoBuffer<npy_intp> _sizes(dims + 1);
		for (i = 0; i < dims; i++)
			_sizes[i] = sizes[i];
		if (cn > 1)
			_sizes[dims++] = cn;
		PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
		if (!o)
			CV_Error_(Error::StsError,
			("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
		return allocate(o, dims0, sizes, type, step);
	}

	bool allocate(cv::UMatData* u, int accessFlags,
		cv::UMatUsageFlags usageFlags) const {
		return stdAllocator->allocate(u, accessFlags, usageFlags);
	}

	void deallocate(cv::UMatData* u) const {
		if (u) {
			PyEnsureGIL gil;
			PyObject* o = (PyObject*)u->userdata;
			Py_XDECREF(o);
			delete u;
		}
	}

	const cv::MatAllocator* stdAllocator;
};
enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };
//===================   ALLOCATOR INITIALIZTION   ==================================================
static NumpyAllocator g_numpyAllocator;

//===================   SOME FUNCTION From cv2.hpp  ==================================================
struct ArgInfo
{
	const char * name;
	bool outputarg;
	// more fields may be added if necessary

	ArgInfo(const char * name_, bool outputarg_)
		: name(name_)
		, outputarg(outputarg_) {}

	// to match with older pyopencv_to function signature
	operator const char *() const { return name; }
};
static inline bool pyopencv_to(PyObject* o, cv::Mat& m, const ArgInfo info)
{
	bool allowND = true;
	if (!o || o == Py_None)
	{
		if (!m.data)
			m.allocator = &g_numpyAllocator;
		return true;
	}

	if (PyInt_Check(o))
	{
		double v[] = { static_cast<double>(PyInt_AsLong((PyObject*)o)), 0., 0., 0. };
		m = cv::Mat(4, 1, CV_64F, v).clone();
		return true;
	}
	if (PyFloat_Check(o))
	{
		double v[] = { PyFloat_AsDouble((PyObject*)o), 0., 0., 0. };
		m = cv::Mat(4, 1, CV_64F, v).clone();
		return true;
	}
	if (PyTuple_Check(o))
	{
		int i, sz = (int)PyTuple_Size((PyObject*)o);
		m = cv::Mat(sz, 1, CV_64F);
		for (i = 0; i < sz; i++)
		{
			PyObject* oi = PyTuple_GET_ITEM(o, i);
			if (PyInt_Check(oi))
				m.at<double>(i) = (double)PyInt_AsLong(oi);
			else if (PyFloat_Check(oi))
				m.at<double>(i) = (double)PyFloat_AsDouble(oi);
			else
			{
				failmsg("%s is not a numerical tuple", info.name);
				m.release();
				return false;
			}
		}
		return true;
	}

	if (!PyArray_Check(o))
	{
		failmsg("%s is not a numpy array, neither a scalar", info.name);
		return false;
	}

	PyArrayObject* oarr = (PyArrayObject*)o;

	bool needcopy = false, needcast = false;
	int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
	int type = typenum == NPY_UBYTE ? CV_8U :
		typenum == NPY_BYTE ? CV_8S :
		typenum == NPY_USHORT ? CV_16U :
		typenum == NPY_SHORT ? CV_16S :
		typenum == NPY_INT ? CV_32S :
		typenum == NPY_INT32 ? CV_32S :
		typenum == NPY_FLOAT ? CV_32F :
		typenum == NPY_DOUBLE ? CV_64F : -1;

	if (type < 0)
	{
		if (typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG)
		{
			needcopy = needcast = true;
			new_typenum = NPY_INT;
			type = CV_32S;
		}
		else
		{
			failmsg("%s data type = %d is not supported", info.name, typenum);
			return false;
		}
	}

#ifndef CV_MAX_DIM
	const int CV_MAX_DIM = 32;
#endif

	int ndims = PyArray_NDIM(oarr);
	if (ndims >= CV_MAX_DIM)
	{
		failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
		return false;
	}

	int size[CV_MAX_DIM + 1];
	size_t step[CV_MAX_DIM + 1];
	size_t elemsize = CV_ELEM_SIZE1(type);
	const npy_intp* _sizes = PyArray_DIMS(oarr);
	const npy_intp* _strides = PyArray_STRIDES(oarr);
	bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

	for (int i = ndims - 1; i >= 0 && !needcopy; i--)
	{
		// these checks handle cases of
		//  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
		//  b) transposed arrays, where _strides[] elements go in non-descending order
		//  c) flipped arrays, where some of _strides[] elements are negative
		// the _sizes[i] > 1 is needed to avoid spurious copies when NPY_RELAXED_STRIDES is set
		if ((i == ndims - 1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize) ||
			(i < ndims - 1 && _sizes[i] > 1 && _strides[i] < _strides[i + 1]))
			needcopy = true;
	}

	if (ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2])
		needcopy = true;

	if (needcopy)
	{
		if (info.outputarg)
		{
			failmsg("Layout of the output array %s is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)", info.name);
			return false;
		}

		if (needcast) {
			o = PyArray_Cast(oarr, new_typenum);
			oarr = (PyArrayObject*)o;
		}
		else {
			oarr = PyArray_GETCONTIGUOUS(oarr);
			o = (PyObject*)oarr;
		}

		_strides = PyArray_STRIDES(oarr);
	}

	// Normalize strides in case NPY_RELAXED_STRIDES is set
	size_t default_step = elemsize;
	for (int i = ndims - 1; i >= 0; --i)
	{
		size[i] = (int)_sizes[i];
		if (size[i] > 1)
		{
			step[i] = (size_t)_strides[i];
			default_step = step[i] * size[i];
		}
		else
		{
			step[i] = default_step;
			default_step *= size[i];
		}
	}

	// handle degenerate case
	if (ndims == 0) {
		size[ndims] = 1;
		step[ndims] = elemsize;
		ndims++;
	}

	if (ismultichannel)
	{
		ndims--;
		type |= CV_MAKETYPE(0, size[2]);
	}

	if (ndims > 2 && !allowND)
	{
		failmsg("%s has more than 2 dimensions", info.name);
		return false;
	}

	m = cv::Mat(ndims, size, type, PyArray_DATA(oarr), step);
	m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
	m.addref();

	if (!needcopy)
	{
		Py_INCREF(o);
	}
	m.allocator = &g_numpyAllocator;

	return true;
}
static inline bool pyopencv_to(PyObject* obj, cv::Rect& r, const char* name)
{
	(void)name;
	if (!obj || obj == Py_None)
		return true;
	return PyArg_ParseTuple(obj, "iiii", &r.x, &r.y, &r.width, &r.height) > 0;
}
static inline bool pyopencv_to(PyObject* obj, cv::Point& p, const char* name)
{
	(void)name;
	if (!obj || obj == Py_None)
		return true;
	if (!!PyComplex_CheckExact(obj))
	{
		Py_complex c = PyComplex_AsCComplex(obj);
		p.x = saturate_cast<int>(c.real);
		p.y = saturate_cast<int>(c.imag);
		return true;
	}
	return PyArg_ParseTuple(obj, "ii", &p.x, &p.y) > 0;
}
static inline PyObject* pyopencv_from(const cv::Rect& r)
{
	return Py_BuildValue("(iiii)", r.x, r.y, r.width, r.height);
}
static inline PyObject* pyopencv_from(const cv::Rect2d& r)
{
	return Py_BuildValue("(dddd)", r.x, r.y, r.width, r.height);
}
static inline PyObject* pyopencv_from(const cv::Mat& m)
{
	if (!m.data)
		Py_RETURN_NONE;
	cv::Mat temp, *p = (cv::Mat*)&m;
	if (!p->u || p->allocator != &g_numpyAllocator)
	{
		temp.allocator = &g_numpyAllocator;
		ERRWRAP2(m.copyTo(temp));
		p = &temp;
	}
	PyObject* o = (PyObject*)p->u->userdata;
	Py_INCREF(o);
	return o;
}
static inline PyObject* pyopencv_from(const cv::Point& p)
{
	return Py_BuildValue("(ii)", p.x, p.y);
}
template<typename _Tp> static inline bool pyopencv_to_generic_vec(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
{
	if (!obj || obj == Py_None)
		return true;
	if (!PySequence_Check(obj))
		return false;
	PyObject *seq = PySequence_Fast(obj, info.name);
	if (seq == NULL)
		return false;
	int i, n = (int)PySequence_Fast_GET_SIZE(seq);
	value.resize(n);

	PyObject** items = PySequence_Fast_ITEMS(seq);

	for (i = 0; i < n; i++)
	{
		PyObject* item = items[i];
		if (!pyopencv_to(item, value[i], info))
			break;
	}
	Py_DECREF(seq);
	return i == n;
}
template<typename _Tp> static inline PyObject* pyopencv_from_generic_vec(const std::vector<_Tp>& value)
{
	int i, n = (int)value.size();
	PyObject* seq = PyList_New(n);
	for (i = 0; i < n; i++)
	{
		PyObject* item = pyopencv_from(value[i]);
		if (!item)
			break;
		PyList_SET_ITEM(seq, i, item);
	}
	if (i < n)
	{
		Py_DECREF(seq);
		return 0;
	}
	return seq;
}
template<typename _Tp> static inline PyObject* pyopencv_from_generic_vec_point(const std::vector<_Tp>& value)
{
	int i, n = (int)value.size();
	PyObject* seq = PyList_New(n);
	for (i = 0; i < n; i++)
	{
		PyObject* item = pyopencv_from(value[i]);
		if (!item)
			break;
		PyList_SET_ITEM(seq, i, item);
	}
	if (i < n)
	{
		Py_DECREF(seq);
		return 0;
	}
	return seq;
}
template<typename _Tp> struct pyopencvVecConverter
{
	static bool to(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
	{
		typedef typename DataType<_Tp>::channel_type _Cp;
		if (!obj || obj == Py_None)
			return true;
		if (PyArray_Check(obj))
		{
			Mat m;
			pyopencv_to(obj, m, info);
			m.copyTo(value);
		}
		if (!PySequence_Check(obj))
			return false;
		PyObject *seq = PySequence_Fast(obj, info.name);
		if (seq == NULL)
			return false;
		int i, j, n = (int)PySequence_Fast_GET_SIZE(seq);
		value.resize(n);

		int type = traits::Type<_Tp>::value;
		int depth = CV_MAT_DEPTH(type), channels = CV_MAT_CN(type);
		PyObject** items = PySequence_Fast_ITEMS(seq);

		for (i = 0; i < n; i++)
		{
			PyObject* item = items[i];
			PyObject* seq_i = 0;
			PyObject** items_i = &item;
			_Cp* data = (_Cp*)&value[i];

			if (channels == 2 && PyComplex_CheckExact(item))
			{
				Py_complex c = PyComplex_AsCComplex(obj);
				data[0] = saturate_cast<_Cp>(c.real);
				data[1] = saturate_cast<_Cp>(c.imag);
				continue;
			}
			if (channels > 1)
			{
				if (PyArray_Check(item))
				{
					Mat src;
					pyopencv_to(item, src, info);
					if (src.dims != 2 || src.channels() != 1 ||
						((src.cols != 1 || src.rows != channels) &&
						(src.cols != channels || src.rows != 1)))
						break;
					Mat dst(src.rows, src.cols, depth, data);
					src.convertTo(dst, type);
					if (dst.data != (uchar*)data)
						break;
					continue;
				}

				seq_i = PySequence_Fast(item, info.name);
				if (!seq_i || (int)PySequence_Fast_GET_SIZE(seq_i) != channels)
				{
					Py_XDECREF(seq_i);
					break;
				}
				items_i = PySequence_Fast_ITEMS(seq_i);
			}

			for (j = 0; j < channels; j++)
			{
				PyObject* item_ij = items_i[j];
				if (PyInt_Check(item_ij))
				{
					int v = (int)PyInt_AsLong(item_ij);
					if (v == -1 && PyErr_Occurred())
						break;
					data[j] = saturate_cast<_Cp>(v);
				}
				else if (PyLong_Check(item_ij))
				{
					int v = (int)PyLong_AsLong(item_ij);
					if (v == -1 && PyErr_Occurred())
						break;
					data[j] = saturate_cast<_Cp>(v);
				}
				else if (PyFloat_Check(item_ij))
				{
					double v = PyFloat_AsDouble(item_ij);
					if (PyErr_Occurred())
						break;
					data[j] = saturate_cast<_Cp>(v);
				}
				else
					break;
			}
			Py_XDECREF(seq_i);
			if (j < channels)
				break;
		}
		Py_DECREF(seq);
		return i == n;
	}

	static PyObject* from(const std::vector<_Tp>& value)
	{
		if (value.empty())
			return PyTuple_New(0);
		int type = traits::Type<_Tp>::value;
		int depth = CV_MAT_DEPTH(type), channels = CV_MAT_CN(type);
		cv::Mat src((int)value.size(), channels, depth, (uchar*)&value[0]);
		return pyopencv_from(src);
	}
};

//===================   CONVERSION CLASS DEFINE   ==================================================
// static void init()
// {
// 	import_array();
// }
class NDArrayConverter
{
private:
	//调用python初始化import接口
	void init()
	{
		import_array();
	}
public:
	//类构造函数，初始化python的import
	NDArrayConverter() 
	{ init(); };
	//python对象转mat
	cv::Mat toMat(PyObject *o)
	{
		cv::Mat m;
		bool allowND = true;
		if (!PyArray_Check(o)) {
			failmsg("argument is not a numpy array");
			if (!m.data)
				m.allocator = &g_numpyAllocator;
		}
		else {
			PyArrayObject* oarr = (PyArrayObject*)o;

			bool needcopy = false, needcast = false;
			int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
			int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
				typenum == NPY_USHORT ? CV_16U :
				typenum == NPY_SHORT ? CV_16S :
				typenum == NPY_INT ? CV_32S :
				typenum == NPY_INT32 ? CV_32S :
				typenum == NPY_FLOAT ? CV_32F :
				typenum == NPY_DOUBLE ? CV_64F : -1;

			if (type < 0) {
				if (typenum == NPY_INT64 || typenum == NPY_UINT64
					|| type == NPY_LONG) {
					needcopy = needcast = true;
					new_typenum = NPY_INT;
					type = CV_32S;
				}
				else {
					failmsg("Argument data type is not supported");
					m.allocator = &g_numpyAllocator;
					return m;
				}
			}

#ifndef CV_MAX_DIM
			const int CV_MAX_DIM = 32;
#endif

			int ndims = PyArray_NDIM(oarr);
			if (ndims >= CV_MAX_DIM) {
				failmsg("Dimensionality of argument is too high");
				if (!m.data)
					m.allocator = &g_numpyAllocator;
				return m;
			}

			int size[CV_MAX_DIM + 1];
			size_t step[CV_MAX_DIM + 1];
			size_t elemsize = CV_ELEM_SIZE1(type);
			const npy_intp* _sizes = PyArray_DIMS(oarr);
			const npy_intp* _strides = PyArray_STRIDES(oarr);
			bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

			for (int i = ndims - 1; i >= 0 && !needcopy; i--) {
				// these checks handle cases of
				//  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
				//  b) transposed arrays, where _strides[] elements go in non-descending order
				//  c) flipped arrays, where some of _strides[] elements are negative
				if ((i == ndims - 1 && (size_t)_strides[i] != elemsize)
					|| (i < ndims - 1 && _strides[i] < _strides[i + 1]))
					needcopy = true;
			}

			if (ismultichannel && _strides[1] != (npy_intp)elemsize * _sizes[2])
				needcopy = true;

			if (needcopy) {

				if (needcast) {
					o = PyArray_Cast(oarr, new_typenum);
					oarr = (PyArrayObject*)o;
				}
				else {
					oarr = PyArray_GETCONTIGUOUS(oarr);
					o = (PyObject*)oarr;
				}

				_strides = PyArray_STRIDES(oarr);
			}

			for (int i = 0; i < ndims; i++) {
				size[i] = (int)_sizes[i];
				step[i] = (size_t)_strides[i];
			}

			// handle degenerate case
			if (ndims == 0) {
				size[ndims] = 1;
				step[ndims] = elemsize;
				ndims++;
			}

			if (ismultichannel) {
				ndims--;
				type |= CV_MAKETYPE(0, size[2]);
			}

			if (ndims > 2 && !allowND) {
				failmsg("%s has more than 2 dimensions");
			}
			else {

				m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
				m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
				m.addref();

				if (!needcopy) {
					Py_INCREF(o);
				}
			}
			m.allocator = &g_numpyAllocator;
		}
		return m;
	}
	//mat对象转python
	PyObject* toNDArray(const cv::Mat& m)
	{
		if (!m.data)
			Py_RETURN_NONE;
		Mat temp, *p = (Mat*)&m;
		if (!p->u || p->allocator != &g_numpyAllocator) {
			temp.allocator = &g_numpyAllocator;
			ERRWRAP2(m.copyTo(temp));
			p = &temp;
		}
		PyObject* o = (PyObject*)p->u->userdata;
		Py_INCREF(o);
		return o;
	}

	//vector类型互转模板
	//python数据转换成C++的vector
	
	template<typename _Tp>
	bool pyopencv_vec_to(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
	{
		return pyopencvVecConverter<_Tp>::to(obj, value, info);
	}
	//C++中vector<int>、vector<double>类型转Pyobject*
	template<typename _Tp>
	PyObject* pyopencv_vec_from(const std::vector<_Tp>& value)
	{
		return pyopencvVecConverter<_Tp>::from(value);
	}
	//C++中vector<Rect>类型转Pyobject*
	template<typename _Tp>
	PyObject* pyopencv_vec_from_Rect(const std::vector<_Tp>& value)
	{
		return pyopencv_from_generic_vec(value);
	}
	//C++中vector<Point>类型转Pyobject*
	template<typename _Tp>
	PyObject* pyopencv_vec_from_point(const std::vector<_Tp>& value)
	{
		return pyopencv_from_generic_vec_point(value);
	}
	
	//Point类型C++转Python
	PyObject* pyopencv_from_point(const cv::Point& p)
	{
		return pyopencv_from(p);
	}
	//Point类型Python转C++
	bool pyopencv_to_point(PyObject* obj, cv::Point& p, const char* name)
	{
		return pyopencv_to(obj, p, name);
	}
};

//===================   CONVERSION CLASS FUNCTION   ==================================================

# endif
