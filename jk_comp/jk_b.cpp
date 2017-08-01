#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <lawrap/blas.h>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <omp.h>

namespace py = pybind11;


inline int index(int a, int b)
{
    return (a>b ? a*(a+1)/2 + b : b*(b+1)/2 + a);
}

std::vector<py::array> make_J(py::array_t<double> g, py::array_t<double> D)
{
    int ij, kl, ik, jl, ikjl, ijkl;
    py::buffer_info g_info = g.request();
    py::buffer_info D_info = D.request();

    const double * g_data = static_cast<double *>(g_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);

    size_t nbf = g_info.shape[0];

    double eri_size = ((nbf * (nbf+1) ) * 0.5 * ((nbf * (nbf+1) ) * 0.5 + 1)) * 0.5;

    std::vector<double> eri(eri_size);

    // Access nbf^3 * i + nbf^2 * j + nbf * k + l
    
    std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;
        
    double start = omp_get_wtime();

#pragma omp parallel for private(ij,kl,ijkl) schedule(dynamic)
    for(size_t i = 0; i < nbf; i++)
    {
        for(size_t j = 0; j <= i; j++)
        {
            ij = index(i,j);
            for(size_t k = 0; k < nbf; k++)
            {
                for(size_t l = 0; l <= k; l++)
                {
                    kl = index(k,l);
                    ijkl = index(ij,kl);
                    eri[ijkl] = g_data[nbf*nbf*nbf*i + nbf*nbf*j + nbf*k + l];
                }
            }
        }
    }

    std::vector<double> J_data(nbf * nbf);
    std::vector<double> K_data(nbf * nbf);

#pragma omp parallel private(ij,ik,jl,kl,ijkl,ikjl)
{
    std::vector<double> tempJ(nbf*nbf);
    std::vector<double> tempK(nbf*nbf);

    #pragma omp for schedule(dynamic)
    for(size_t i = 0; i < nbf; i++)
    {
        for(size_t j = 0; j <= i; j++)
        {
//Creating J
            ij = index(i,j);
            for(size_t k = 0; k < nbf; k++)
            {
                for(size_t l = 0; l < nbf; l++)
                {
                    kl = index(k,l);
                    ik = index(i,k);
                    jl = index(j,l);
                    ijkl = index(ij,kl);
                    ikjl = index(ik,jl);
                    tempJ[k*nbf + l] = eri[ijkl];
                    tempK[k*nbf + l] = eri[ikjl];
                }
            }
            J_data[i * nbf + j] = LAWrap::dot(nbf*nbf, tempJ.data(), 1, D_data, 1);
            J_data[j * nbf + i] = J_data [ i * nbf + j];
            K_data[i * nbf + j] = LAWrap::dot(nbf*nbf, tempK.data(), 1, D_data, 1);
            K_data[j * nbf + i] = K_data [ i * nbf + j];
        }
    }
}

    py::buffer_info Jbuff =
        {
            J_data.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            { nbf, nbf },
            { nbf * sizeof(double), sizeof(double) }
        };

    py::buffer_info Kbuff =
        {
            K_data.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            { nbf, nbf },
            { nbf * sizeof(double), sizeof(double) }
        };


    py::array J(Jbuff);
    py::array K(Kbuff);
    double stop = omp_get_wtime();
    std::cout << "Comp. time: " <<  (stop - start) << std::endl;
    return {J, K};
}


PYBIND11_PLUGIN(jk_mod)
{
    py::module jk("jk_mod", "JK module");

    jk.def("make_J", &make_J, "God I hope this makes J");

return jk.ptr();
}
