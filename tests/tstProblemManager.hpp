#ifndef _TSTPROBLEMMANGER_HPP_
#define _TSTPROBLEMMANGER_HPP_

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>
#include <ProblemManager.hpp>

#include <mpi.h>

#include "tstMesh.hpp"

template <class T> 
class ProblemManagerTest : public MeshTest<T> {
  protected:
    void SetUp() {
        MeshTest<T>::SetUp();
    }

    void TearDown() {
        MeshTest<T>::TearDown();
    }
};

#endif // _TSTPROBLEMMANAGER_HPP_
