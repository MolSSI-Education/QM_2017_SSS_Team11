#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <iostream>
#include <pybind11/numpy.h>
#include <lawrap/blas.h>

namespace py = pybind11;

py::array_t<double> calc_J (py::array_t<double> g,
                     py::array_t<double> D)
{

    py::buffer_info g_info = g.request();
    py::buffer_info D_info = D.request();

    size_t g_nrows = g_info.shape[0];
    size_t g_ncols = g_info.shape[1];
    size_t D_k = D_info.shape[0];

    const double * g_data = static_cast<double *>(g_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);

    std::vector<double> J_data(g_nrows);

    LAWrap::gemv('N',g_nrows,g_ncols,1.0,g_data,g_nrows,D_data,1,0.0,J_data.data(),1);

    py::buffer_info Jbuf =
        {
            J_data.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            1,
            { g_ncols },
            { sizeof(double) }
        };

    return py::array_t<double>(Jbuf);
}

PYBIND11_PLUGIN(jk_blas)
{
    py::module m("jk_blas", "Ben's basic module");
    m.def("calc_J", &calc_J, "get J matrix");
    return m.ptr();
}
