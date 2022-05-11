#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>
#include <ProblemManager.hpp>

#include <mpi.h>

#include "tstDriver.hpp"
#include "tstMesh.hpp"
#include "tstProblemManager.hpp"

TYPED_TEST_SUITE( ProblemManagerTest, MeshDeviceTypes);

using Cell = Cajita::Cell;
using FaceI = Cajita::Face<Cajita::Dim::I>;
using FaceJ = Cajita::Face<Cajita::Dim::J>;
using FaceK = Cajita::Face<Cajita::Dim::K>;
using Quantity = CajitaFluids::Field::Quantity;
using Velocity = CajitaFluids::Field::Velocity;
using Current = CajitaFluids::Version::Current;
using Next = CajitaFluids::Version::Next;

TYPED_TEST(ProblemManagerTest, StateArrayTest)
{
    using ExecutionSpace = typename TestFixture::ExecutionSpace;

    // Get basic mesh state
    auto pm = this->testPM_;
    auto mesh = pm->mesh();
    auto qcurr = pm->get(Cell(), Quantity(), Current());
    auto ucurr = pm->get(FaceI(), Velocity(), Current());
    auto vcurr = pm->get(FaceJ(), Velocity(), Current());
    auto qnext = pm->get(Cell(), Quantity(), Next());
    auto unext = pm->get(FaceI(), Velocity(), Next());
    auto vnext = pm->get(FaceJ(), Velocity(), Next());
    auto rank = mesh->rank();

    /* Set values in the array based on our rank. Each cell gets a value of 
     * rank*1000 + i * 100 + j * 10 + an int based on its type as follows
     * FaceI: +1
     * FaceJ: +2
     * Next(): +5
     */
    auto cell_space = mesh->localGrid()->indexSpace(Cajita::Own(), Cell(), Cajita ::Local());
    auto u_space = mesh->localGrid()->indexSpace(Cajita::Own(), FaceI(), Cajita ::Local());
    auto v_space = mesh->localGrid()->indexSpace(Cajita::Own(), FaceJ(), Cajita ::Local());
    Kokkos::parallel_for(
        "InitializeCellFields", 
        createExecutionPolicy(cell_space, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            qcurr(i, j, 0) = rank * 1000 + i * 100 + j * 10;
            qnext(i, j, 0) = rank * 1000 + i * 100 + j * 10 + 5;
        });
    Kokkos::parallel_for(
        "InitializeFaceIFields", 
        createExecutionPolicy(v_space, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            ucurr(i, j, 0) = rank * 1000 + i * 100 + j * 10 + 1;
            unext(i, j, 0) = rank * 1000 + i * 100 + j * 10 + 5 + 1;
        });
    Kokkos::parallel_for(
        "InitializeFaceJFields", 
        createExecutionPolicy(u_space, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            vcurr(i, j, 0) = rank * 1000 + i * 100 + j * 10 + 2;
            vnext(i, j, 0) = rank * 1000 + i * 100 + j * 10 + 5 + 2;
        });

   // Check that we can swap the views properly

   // Check that we can halo the views appropriately

   // ASSERT_EQ(...);
}

TYPED_TEST(ProblemManagerTest, HaloTest)
{
}
