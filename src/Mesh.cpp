/****************************************************************************
 * Copyright (c) 2021 by the CajitaFluids authors                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CajitaFluids benchmark. CajitaFluids is         *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Mesh.hpp>

namespace CajitaFluids
{
//---------------------------------------------------------------------------//
template class Mesh<2, Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class Mesh<2, Kokkos::CudaSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace CajitaFluids
