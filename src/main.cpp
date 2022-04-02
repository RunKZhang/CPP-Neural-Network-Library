#include <pybind11/pybind11.h>
#include "cmatrix.h"

namespace py = pybind11;

int add(int a, int b)
{
  return a+b;
}

PYBIND11_MODULE(mymath,m){

     m.def("add", &add);

     py::class_<CMatrix>(m, "CMatrix", py::buffer_protocol())
           .def(py::init<ssize_t, ssize_t>())
           .def_buffer([](CMatrix &m) -> py::buffer_info {
       return py::buffer_info(
                   m.data(),                               /* Pointer to buffer */
                   sizeof(float),                          /* Size of one scalar */
                   py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                   2,                                      /* Number of dimensions */
       { m.rows(), m.cols() },                 /* Buffer dimensions */
       { sizeof(float) * m.cols(),             /* Strides (in bytes) for each index */
         sizeof(float) }
                   );
   })
   .def(py::init([](py::buffer const b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<float>::format() || info.ndim != 2)
            throw std::runtime_error("Incompatible buffer format!");

            auto v = new CMatrix(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(float) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

           .def("rows", &CMatrix::rows)
           .def("cols", &CMatrix::cols)

           /// Bare bones interface
           .def("__getitem__", [](const CMatrix &m, std::pair<ssize_t, ssize_t> i) {
       if (i.first >= m.rows() || i.second >= m.cols())
           throw py::index_error();
       return m(i.first, i.second);
   })
           .def("__setitem__", [](CMatrix &m, std::pair<ssize_t, ssize_t> i, float v) {
       if (i.first >= m.rows() || i.second >= m.cols())
           throw py::index_error();
       m(i.first, i.second) = v;
   })
   ;

}
  
