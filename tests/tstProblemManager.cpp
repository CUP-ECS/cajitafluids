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
    auto qnext = pm->get(Cell(), Quantity(), Next());
    auto rank = mesh->rank();

    /* Set values in the array based on our rank. Each cell gets a value of 
     * rank*1000 + i * 100 + j * 10 + an int based on its type as follows
     * FaceI: +1
     * FaceJ: +2
     * Next(): +5
     */
    auto qspace = mesh->localGrid()->indexSpace(Cajita::Own(), Cell(), Cajita ::Local());
    Kokkos::parallel_for(
        "InitializeCellFields", 
        createExecutionPolicy(qspace, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            qcurr(i, j, 0) = rank * 1000 + i * 100 + j * 10;
            qnext(i, j, 0) = rank * 1000 + i * 100 + j * 10 + 5;
        });

    // Check that we can swap the views properly by checking the cell view
    // (we don't check the other views for now)
    pm->advance(Cell(), Quantity());
    auto qcurr = pm->get(Cell(), Quantity(), Current());
    auto qcopy = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), 
        qcurr);
    for (int i = qspace.min(0); i < qspace.max(0); i++) 
        for (int j = qspace.min(1); j < qspace.max(1); j++) 
            ASSERT_EQ(qcopy(i, j, 0), rank * 1000 + i * 100 + j * 10 + 5);
}

TYPED_TEST(ProblemManagerTest, HaloTest)
{
    using ExecutionSpace = typename TestFixture::ExecutionSpace;

    auto pm = this->testPM_;
    auto mesh = pm->mesh();
    auto rank = mesh->rank();
    auto ucurr = pm->get(FaceI(), Velocity(), Current());
    auto uspace = mesh->localGrid()->indexSpace(Cajita::Own(), FaceI(), Cajita ::Local());
    Kokkos::parallel_for(
        "InitializeFaceIFields", 
        createExecutionPolicy(uspace, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            ucurr(i, j, 0) = rank * 1000 + i * 100 + j * 10 + 1;
        });

    // Check that we can halo the views appropriately, using the FaceI direction
    // as the check
    pm->gather( Current() );
    std::array<int,2> directions[4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    auto ucopy = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ucurr);
    for (int i = 0; i < 4; i++) 
    {
        auto dir = directions[i];
        int neighbor_rank = mesh->localGrid()->neighborRank(dir);
        auto u_shared_space = mesh->localGrid()->sharedIndexSpace(Cajita::Ghost(), 
            FaceI(), dir);
        for (int i = u_shared_space.min(0); i < u_shared_space.max(0); i++) 
            for (int j = u_shared_space.min(1); j < u_shared_space.max(1); j++)
                ASSERT_EQ(ucopy(i, j, 0), neighbor_rank * 1000 + i * 100 + j * 10 + 1);
    }
}
