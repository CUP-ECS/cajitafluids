#ifndef _TSTDRIVER_HPP_
#define _TSTDRIVER_HPP_

#include "gtest/gtest.h"

#include <Kokkos_Core.hpp>

template <class E, class M>
struct DeviceType
{
  using ExecutionSpace = E;
  using MemorySpace = M;
};

using MeshDeviceTypes = ::testing::Types<
#ifdef KOKKOS_ENABLE_OPENMP
    DeviceType<Kokkos::OpenMP, Kokkos::HostSpace>,
#endif
#ifdef KOKKOS_ENABLE_CUDA
    DeviceType<Kokkos::Cuda, Kokkos::CudaSpace>,
#endif
    DeviceType<Kokkos::Serial, Kokkos::HostSpace> >;

#endif //_TSTDRIVER_HPP_
