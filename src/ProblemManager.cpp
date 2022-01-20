/****************************************************************************
 * Copyright (c) 2018-2020 by the CajitaFluids authors                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CajitaFluids benchmark. CajitaFluids is         *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ProblemManager.hpp>
#include <Solver.hpp>

namespace CajitaFluids
{
//---------------------------------------------------------------------------//
template class ProblemManager<2, Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>;

//---------------------------------------------------------------------------//

} // end namespace CajitaFluids
