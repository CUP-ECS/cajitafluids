/****************************************************************************
 * Copyright (c) 2018-2020 by the CabandFluids authors                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CabandFluids benchmark. CabanaFluids is         *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Mesh.hpp>

namespace CabanaFluids
{
//---------------------------------------------------------------------------//
template class Mesh<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class Mesh<Kokkos::CudaSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace CabanaFluids
