/****************************************************************************
 * Copyright (c) 2018-2020 by the CajitaFluids authors                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CajitaFluids library. CajitaFluids is           *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITAFLUIDS_SOLVER_HPP
#define CAJITAFLUIDS_SOLVER_HPP

#include <Cajita_HypreStructuredSolver.hpp>
#include <Cajita_Partitioner.hpp>
#include <Cajita_ReferenceStructuredSolver.hpp>
#include <Cajita_Types.hpp>

#include <BodyForce.hpp>
#include <BoundaryConditions.hpp>
#include <InflowSource.hpp>
#include <Mesh.hpp>
#include <ProblemManager.hpp>
#include <SiloWriter.hpp>
#include <TimeIntegrator.hpp>
#include <VelocityCorrector.hpp>

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

#include <mpi.h>

namespace CajitaFluids
{
/*
 * Convenience base class so that examples that use this don't need to know
 * the details of the problem manager/mesh/etc templating.
 */
class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void setup( void ) = 0;
    virtual void step( void ) = 0;
    virtual void solve( const double t_final, const int write_freq ) = 0;
};

template <std::size_t NumSpaceDim, class ExecutionSpace, class MemorySpace>
class Solver;

//---------------------------------------------------------------------------//
template <class ExecutionSpace, class MemorySpace>
class Solver<2, ExecutionSpace, MemorySpace> : public SolverBase
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cajita::UniformMesh<double, 2>;
    using cell_array =
        Cajita::Array<double, Cajita::Cell, mesh_type, MemorySpace>;
    using pm_type = ProblemManager<2, ExecutionSpace, MemorySpace>;
    using bc_type = BoundaryCondition<2>;

    using Cell = Cajita::Cell;
    using FaceI = Cajita::Face<Cajita::Dim::I>;
    using FaceJ = Cajita::Face<Cajita::Dim::J>;

    template <class InitFunc>
    Solver( MPI_Comm comm, const Kokkos::Array<double, 4>& global_bounding_box,
            const std::array<int, 2>& global_num_cell,
            const Cajita::BlockPartitioner<2>& partitioner,
            const double density, const InitFunc& create_functor,
            const BoundaryCondition<2>& bc, const InflowSource<2>& source,
            const BodyForce<2>& body, const double delta_t,
            const std::string& matrix_solver,
            const std::string& preconditioner )
        : _halo_min( 3 )
        , _density( density )
        , _bc( bc )
        , _source( source )
        , _body( body )
        , _dt( delta_t )
        , _time( 0.0 )
    {

        // Create a mesh one which to do the solve and a problem manager to
        // handle state
        _mesh = std::make_shared<Mesh<2, ExecutionSpace, MemorySpace>>(
            global_bounding_box, global_num_cell, partitioner, _halo_min,
            comm );

        // Check that our timestep is small enough to handle the mesh size
        // given and inflow/body forces under some very simple assumptions.
        // In the end, the user should give us sane input conditions.
        auto forcemax = sqrt( body._force[0] * body._force[0] +
                              body._force[1] * body._force[1] );
        auto umax =
            fmax( fabs( source._velocity[0] ), fabs( source._velocity[1] ) ) +
            sqrt( forcemax * _mesh->cellSize() );
        if ( ( umax > 0 ) && ( _dt > _mesh->cellSize() / umax ) )
        {
            _dt = _mesh->cellSize() / umax;
            std::cerr << "Reducting timestep to " << _dt
                      << " given mesh size and inflow velocity.\n";
        }

        // Put domain rane information into the boundary condition object
        _bc.min = _mesh->minDomainGlobalCellIndex();
        _bc.max = _mesh->maxDomainGlobalCellIndex();

        // Create a problem manager to manage mesh state
        _pm = std::make_shared<ProblemManager<2, ExecutionSpace, MemorySpace>>(
            _mesh, create_functor );

        // Create a velocity corrector to enforce incompressibility
        _vc = createVelocityCorrector<2, ExecutionSpace, MemorySpace>(
            _pm, _bc, _density, _dt, matrix_solver, preconditioner );

        // Set up Silo for I/O
        _silo =
            std::make_shared<SiloWriter<2, ExecutionSpace, MemorySpace>>( _pm );
    }

    void setup() override
    {
        // Should assert taht _time == 0 here.

        // Finish set up of the initial state of the velocity field
        // by adding input and applying hte pressure correction.
        _addInputs();
        _vc->correctVelocity();
    }

    void step() override
    {
        // 1. Advect the quantities forward a time step in the
        // computed velocity field
        TimeIntegrator::step<2>( ExecutionSpace(), *_pm, _dt, _bc );

        // 2. Add any new inflows and body forces.
        _addInputs();

        // 3. Adjust the velocity field to be divergence-free
        _vc->correctVelocity();
        _time += _dt;
    }

    void solve( const double t_final, const int write_freq ) override
    {
        int t = 0;
        int num_step;

        Kokkos::Profiling::pushRegion( "Solve" );

        _silo->siloWrite( strdup( "Mesh" ), t, _time, _dt );
        Kokkos::Profiling::popRegion();

        num_step = t_final / _dt;

        setup();

        // Now start advancing time.
        do
        {
            if ( 0 == _mesh->rank() && 0 == t % write_freq )
                printf( "Step %d / %d at time = %f\n", t, num_step, _time );

            step();

            // 4. Output mesh state periodically
            if ( 0 == t % write_freq )
            {
                _silo->siloWrite( strdup( "Mesh" ), t, _time, _dt );
            }
            t++;
        } while ( ( _time < t_final ) );
    }

    /* Internal methods for the solver - still technically public because Kokkos
     * requires them to be. */
    void _addInputs()
    {
        auto local_grid = *( _mesh->localGrid() );
        auto local_mesh = *( _mesh->localMesh() );
        double cell_size = _mesh->cellSize();
        double cell_area = cell_size * cell_size;

        auto owned_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Cell(),
                                                  Cajita::Local() );

        // Create local variable versions of the class members to avoid needing
        // to use a (potentially expensive) class lambda
        const BoundaryCondition<2>& bc = _bc;
        const InflowSource<2>& source = _source;
        const BodyForce<2>& body = _body;
        const auto delta_t = _dt;

        Kokkos::Profiling::pushRegion( "Solve::AddInputs::Cell" );
        auto quantity =
            _pm->get( Cell(), Field::Quantity(), Version::Current() );
        Kokkos::parallel_for(
            "add cell quantity",
            createExecutionPolicy( owned_cells, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                int idx[2] = { i, j };
                double loc[2];
                local_mesh.coordinates( Cell(), idx, loc );
                double x = loc[0], y = loc[1];

                source( Cajita::Cell(), quantity, i, j, x, y, delta_t,
                        cell_area );
                body( Cajita::Cell(), quantity, i, j, x, y, delta_t,
                      cell_area );
            } );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "Solve::AddInputs::FaceI" );
        auto owned_ifaces =
            local_grid.indexSpace( Cajita::Own(), FaceI(), Cajita::Local() );
        auto ui = _pm->get( FaceI(), Field::Velocity(), Version::Current() );
        auto l2g_facei =
            Cajita::IndexConversion::createL2G( local_grid, FaceI() );

        Kokkos::parallel_for(
            "add external x velocity",
            createExecutionPolicy( owned_ifaces, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                int idx[2] = { i, j };
                int gi, gj;
                double loc[2];
                local_mesh.coordinates( FaceI(), idx, loc );
                double x = loc[0], y = loc[1];

                l2g_facei( i, j, gi, gj );
                source( FaceI(), ui, i, j, x, y, delta_t, cell_area );
                body( FaceI(), ui, i, j, x, y, delta_t, cell_area );
                bc( FaceI(), ui, gi, gj, i, j );
            } );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "Solve::AddInputs::FaceJ" );
        auto owned_jfaces =
            local_grid.indexSpace( Cajita::Own(), FaceJ(), Cajita::Local() );
        auto uj = _pm->get( FaceJ(), Field::Velocity(), Version::Current() );
        auto l2g_facej =
            Cajita::IndexConversion::createL2G( local_grid, FaceJ() );
        Kokkos::parallel_for(
            "add external y velocity",
            createExecutionPolicy( owned_jfaces, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                int idx[2] = { i, j };
                int gi, gj;
                double loc[2];
                local_mesh.coordinates( FaceJ(), idx, loc );
                double x = loc[0], y = loc[1];

                l2g_facej( i, j, gi, gj );
                source( FaceJ(), uj, i, j, x, y, delta_t, cell_area );
                body( FaceJ(), uj, i, j, x, y, delta_t, cell_area );
                bc( FaceJ(), uj, gi, gj, i, j );
            } );
        Kokkos::Profiling::popRegion();
    }

  private:
    /* Solver state variables */
    int _halo_min;
    double _density;
    BoundaryCondition<2> _bc;
    InflowSource<2> _source;
    BodyForce<2> _body;
    double _dt;
    double _time;
    std::shared_ptr<Mesh<2, ExecutionSpace, MemorySpace>> _mesh;
    std::shared_ptr<ProblemManager<2, ExecutionSpace, MemorySpace>> _pm;
    std::shared_ptr<VelocityCorrectorBase> _vc;
    std::shared_ptr<SiloWriter<2, ExecutionSpace, MemorySpace>> _silo;
    int _rank;
};

//---------------------------------------------------------------------------//
// Creation method.
template <class InitFunc>
std::shared_ptr<SolverBase>
createSolver( const std::string& device, MPI_Comm comm,
              const Kokkos::Array<double, 4>& global_bounding_box,
              const std::array<int, 2>& global_num_cell,
              const Cajita::BlockPartitioner<2>& partitioner,
              const double density, const InitFunc& create_functor,
              const BoundaryCondition<2>& bc, const InflowSource<2>& source,
              const BodyForce<2>& body, const double delta_t,
              const std::string& matrix_solver,
              const std::string& preconditioner )
{
    if ( 0 == device.compare( "serial" ) )
    {
// Hypre with CUDA support breaks support for the serial solver. We'll need
// to set it up to use a different solver in that case
#if defined( KOKKOS_ENABLE_SERIAL ) && !defined( KOKKOS_ENABLE_CUDA )
        return std::make_shared<
            CajitaFluids::Solver<2, Kokkos::Serial, Kokkos::HostSpace>>(
            comm, global_bounding_box, global_num_cell, partitioner, density,
            create_functor, bc, source, body, delta_t, matrix_solver,
            preconditioner );
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "openmp" ) )
    {
#if defined( KOKKOS_ENABLE_OPENMP ) && !defined( KOKKOS_ENABLE_CUDA )
        return std::make_shared<
            CajitaFluids::Solver<2, Kokkos::OpenMP, Kokkos::HostSpace>>(
            comm, global_bounding_box, global_num_cell, partitioner, density,
            create_functor, bc, source, body, delta_t, matrix_solver,
            preconditioner );
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "cuda" ) )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return std::make_shared<
            CajitaFluids::Solver<2, Kokkos::Cuda, Kokkos::CudaSpace>>(
            comm, global_bounding_box, global_num_cell, partitioner, density,
            create_functor, bc, source, body, delta_t, matrix_solver,
            preconditioner );
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "hip" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<CajitaFluids::Solver<
            2, Kokkos : Experimental::HIP, Kokkos::Experimental::HIPSpace>>(
            comm, global_bounding_box, global_num_cell, partitioner, density,
            create_functor, bc, source, body, delta_t, matrix_solver,
            preconditioner );
#else
        throw std::runtime_error( "HIP Backend Not Enabled" );
#endif
    }
    else
    {
        throw std::runtime_error( "invalid backend" );
        return nullptr;
    }
}

//---------------------------------------------------------------------------//

} // end namespace CajitaFluids

#endif // end CAJITAFLUIDS_SOLVER_HPP
