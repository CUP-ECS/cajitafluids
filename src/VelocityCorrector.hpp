/****************************************************************************
 * Copyright (c) 2018-2020 by the CabanaFluids authors                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CabanaFluids library. CabanaFluids is           *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANAFLUIDS_VELOCITYCORRECTOR_HPP
#define CABANAFLUIDS_VELOCITYCORRECTOR_HPP

#include <Kokkos_Core.hpp>
#include <Cabana_Grid.hpp>

#include <BoundaryConditions.hpp>
#include <Interpolation.hpp>
#include <Mesh.hpp>
#include <ProblemManager.hpp>

#include <memory>
#include <string>

#include <mpi.h>

namespace CabanaFluids
{
/*
 * VelocityCorrector uses a virtual base class because the overall structure
 * of the class depends on the underlying solver (hypre-based or reference)
 * being used, and this abstracts that away for the classes that call this
 */
class VelocityCorrectorBase
{
  public:
    virtual ~VelocityCorrectorBase() = default;
    virtual void correctVelocity() = 0;
};

template <std::size_t NumSpaceDim, class ExecutionSpace, class MemorySpace,
          class SparseSolver>
class VelocityCorrector;

//---------------------------------------------------------------------------//

template <class ExecutionSpace, class MemorySpace, class SparseSolver>
class VelocityCorrector<2, ExecutionSpace, MemorySpace, SparseSolver>
    : public VelocityCorrectorBase
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using cell_array =
        Cabana::Grid::Array<double, Cabana::Grid::Cell, mesh_type, MemorySpace>;
    using pm_type = ProblemManager<2, ExecutionSpace, MemorySpace>;
    using bc_type = BoundaryCondition<2>;
    using hypre_solver_type =
        Cabana::Grid::HypreStructuredSolver<double, Cabana::Grid::Cell, MemorySpace>;
    using reference_solver_type =
        Cabana::Grid::ReferenceStructuredSolver<double, Cabana::Grid::Cell, mesh_type,
                                          MemorySpace>;
    using Cell = Cabana::Grid::Cell;
    using FaceI = Cabana::Grid::Face<Cabana::Grid::Dim::I>;
    using FaceJ = Cabana::Grid::Face<Cabana::Grid::Dim::J>;

    VelocityCorrector( const std::shared_ptr<pm_type>& pm, bc_type& bc,
                       const std::shared_ptr<SparseSolver>& pressure_solver,
                       double density, double delta_t )
        : _pm( pm )
        , _bc( bc )
        , _density( density )
        , _dt( delta_t )
        , _pressure_solver( pressure_solver )
    {
        _mesh = pm->mesh();

        // Create the LHS and RHS vectors used calculate pressure correction
        // amount at each cell
        auto vector_layout =
            Cabana::Grid::createArrayLayout( _mesh->localGrid(), 1, Cell() );
        _lhs = Cabana::Grid::createArray<double, MemorySpace>( "pressure LHS",
                                                         vector_layout );
        _rhs = Cabana::Grid::createArray<double, MemorySpace>( "pressure RHS",
                                                         vector_layout );
        Cabana::Grid::ArrayOp::assign( *_lhs, 0.0, Cabana::Grid::Ghost() );
        Cabana::Grid::ArrayOp::assign( *_rhs, 0.0, Cabana::Grid::Ghost() );

        // Set up the solver to compute the pressure at each point using a
        // 5-point 2d laplacian stencil. The matrix itself will have 0 weights
        // for elements that fall outside the boundary, set up by the boundary
        // condition object.
        std::vector<std::array<int, 2>> stencil = {
            { 0, 0 }, { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        _pressure_solver->setMatrixStencil( stencil, false );

        // Fill the associated matrix assocuated with with values
        fillMatrixValues( _pressure_solver );

        _pressure_solver->setTolerance( 1.0e-6 );
        _pressure_solver->setMaxIter( 2000 );
        _pressure_solver->setPrintLevel( 1 );
        _pressure_solver->setup();

        // Finally, we need to halo pressure values with neighbors with whom
        // we share a face so that we can correct velocities on those faces.
        // Note that this is a much simpler and shallower halo poattern than the
        // ones used for advection.
        _pressure_halo = Cabana::Grid::createHalo( Cabana::Grid::FaceHaloPattern<2>(),
                             1, *_lhs );
    }

    template <class View_t>
    void initializeMatrixValues( const View_t& entry_view )
    {
        // Create the matrix entries. The stencil is defined over cells.
        auto local_grid = _mesh->localGrid();

        // Build the solver matrix - set the default entry for each cell
        // then apply the boundary conditions to it
        auto owned_space =
            local_grid->indexSpace( Cabana::Grid::Own(), Cell(), Cabana::Grid::Local() );
        auto l2g = Cabana::Grid::IndexConversion::createL2G( *( _mesh->localGrid() ),
                                                       Cell() );
        auto scale = _dt / ( _density * _mesh->cellSize() * _mesh->cellSize() );
        const bc_type& bc = _bc;

        Kokkos::parallel_for(
            "fill_matrix_entries",
            createExecutionPolicy( owned_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                int gi, gj;
                l2g( i, j, gi, gj );
                entry_view( i, j, 0 ) = 4.0 * scale;
                entry_view( i, j, 1 ) = -1.0 * scale;
                entry_view( i, j, 2 ) = -1.0 * scale;
                entry_view( i, j, 3 ) = -1.0 * scale;
                entry_view( i, j, 4 ) = -1.0 * scale;
                bc.build_matrix( gi, gj, i, j, entry_view, scale );
            } );
    }

    // Specialization of matrix fill for Hypre
    void fillMatrixValues( std::shared_ptr<hypre_solver_type>& solver )
    {
        auto matrix_entry_layout =
            Cabana::Grid::createArrayLayout( _mesh->localGrid(), 5, Cell() );
        auto matrix_entries = Cabana::Grid::createArray<double, MemorySpace>(
            "matrix_entries", matrix_entry_layout );
        initializeMatrixValues( matrix_entries->view() );
        solver->setMatrixValues( *matrix_entries );
    }

    // Specialization of matrix fill for the Cabana structured solver
    void fillMatrixValues( std::shared_ptr<reference_solver_type>& solver )
    {
        const auto& matrix_entries = solver->getMatrixValues();
        auto m = matrix_entries.view();
        initializeMatrixValues( m );

        // Go ahead and set up a simple diagonal preconditioner for the
        // reference solver now
        std::vector<std::array<int, 2>> diag_stencil = { { 0, 0 } };
        solver->setPreconditionerStencil( diag_stencil, false );
        const auto& preconditioner_entries = solver->getPreconditionerValues();
        auto local_grid = _mesh->localGrid();
        auto owned_space =
            local_grid->indexSpace( Cabana::Grid::Own(), Cell(), Cabana::Grid::Local() );
        auto preconditioner_view = preconditioner_entries.view();

        Kokkos::parallel_for(
            "fill_preconditioner_entries",
            createExecutionPolicy( owned_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                preconditioner_view( i, j, 0 ) = 1.0 / m( i, j, 0 );
            } );
    }

    void _buildRHS()
    {
        Kokkos::Profiling::pushRegion(
            "VelocityCorrector::BuildProblem:buildRHS" );

        Kokkos::Profiling::pushRegion(
            "VelocityCorrector::BuildProblem:buildRHS:Gather" );
        // Get the ghosts we need for divergence interpolation
        _pm->gather( Version::Current() );
        Kokkos::Profiling::popRegion();

        Cabana::Grid::ArrayOp::assign( *_rhs, 0.0, Cabana::Grid::Own() );
        auto scale = 1.0 / _mesh->cellSize();

        auto u = _pm->get( FaceI(), Field::Velocity(), Version::Current() );
        auto v = _pm->get( FaceJ(), Field::Velocity(), Version::Current() );

        auto local_grid = _mesh->localGrid();
        auto cell_space = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(),
                                                  Cabana::Grid::Local() );
        auto rhs = _rhs->view();

        Kokkos::parallel_for(
            "ComputeDivergence",
            createExecutionPolicy( cell_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                rhs( i, j, 0 ) = -scale * ( u( i + 1, j, 0 ) - u( i, j, 0 ) +
                                            v( i, j + 1, 0 ) - v( i, j, 0 ) );
            } );
        Kokkos::Profiling::popRegion();
    }

    void _applyPressure()
    {
        Kokkos::Profiling::pushRegion( "VelocityCorrector::ApplyPressure" );
        auto scale = _dt / ( _density * _mesh->cellSize() );

        // Modifies the current velocity field *in place* to be divergence free
        auto u = _pm->get( FaceI(), Field::Velocity(), Version::Current() );
        auto v = _pm->get( FaceJ(), Field::Velocity(), Version::Current() );
        auto p = _lhs->view();

        auto l2g = Cabana::Grid::IndexConversion::createL2G( *( _mesh->localGrid() ),
                                                       Cell() );
        auto local_grid = _mesh->localGrid();
        auto iface_space =
            local_grid->indexSpace( Cabana::Grid::Own(), FaceI(), Cabana::Grid::Local() );
        auto jface_space =
            local_grid->indexSpace( Cabana::Grid::Own(), FaceJ(), Cabana::Grid::Local() );

        /* Now apply the LHS to adjust the velocity field. We need to
         * halo the lhs (the computed pressure) to adjust edge velocities. */
        Kokkos::Profiling::pushRegion(
            "VelocityCorrector::ApplyPressure:Gather" );
        _pressure_halo->gather( ExecutionSpace(), *_lhs );
        Kokkos::Profiling::popRegion();

        const bc_type& bc = _bc;

        Kokkos::parallel_for(
            "PressureUVelocityCorrection",
            createExecutionPolicy( iface_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                u( i, j, 0 ) -= scale * ( p( i, j, 0 ) - p( i - 1, j, 0 ) );

                int gi, gj;
                l2g( i, j, gi, gj );
                bc( FaceI(), u, gi, gj, i, j );
            } );

        Kokkos::parallel_for(
            "PressureVVelocityCorrection",
            createExecutionPolicy( jface_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                v( i, j, 0 ) -= scale * ( p( i, j, 0 ) - p( i, j - 1, 0 ) );

                int gi, gj;
                l2g( i, j, gi, gj );
                bc( FaceJ(), u, gi, gj, i, j );
            } );

        Kokkos::Profiling::popRegion();
    }

    void correctVelocity()
    {
        Kokkos::Profiling::pushRegion( "VelocityCorrector" );

        Kokkos::Profiling::pushRegion( "VelocityCorrector::BuildProblem" );
        _buildRHS();
        Cabana::Grid::ArrayOp::assign( *_lhs, 0.0, Cabana::Grid::Own() );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "VelocityCorrector::PresureSolve" );
        _pressure_solver->solve( *_rhs, *_lhs );
        Kokkos::Profiling::popRegion();

        _applyPressure();

        Kokkos::Profiling::popRegion();
    }

  private:
    std::shared_ptr<pm_type> _pm;
    BoundaryCondition<2> _bc;
    double _density;
    double _dt;
    std::shared_ptr<Mesh<2, ExecutionSpace, MemorySpace>> _mesh;
    std::shared_ptr<SparseSolver> _pressure_solver;
    std::shared_ptr<Cabana::Grid::Halo<MemorySpace>> _pressure_halo;

    std::shared_ptr<cell_array> _lhs;
    std::shared_ptr<cell_array> _rhs;
};

template <std::size_t NumSpaceDims, class ExecutionSpace, class MemorySpace,
          class ProblemManagerType, class BoundaryConditionType>
std::shared_ptr<VelocityCorrectorBase>
createVelocityCorrector( const std::shared_ptr<ProblemManagerType>& pm,
                         BoundaryConditionType& bc, const double density,
                         const double delta_t, std::string solver,
                         std::string precon )
{
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using hypre_solver_type =
        Cabana::Grid::HypreStructuredSolver<double, Cabana::Grid::Cell, MemorySpace>;
    using reference_solver_type =
        Cabana::Grid::ReferenceStructuredSolver<double, Cabana::Grid::Cell, mesh_type,
                                          MemorySpace>;

    auto vector_layout =
        Cabana::Grid::createArrayLayout( pm->mesh()->localGrid(), 1, Cabana::Grid::Cell() );
    if ( solver.compare( "Reference" ) == 0 )
    {
        auto ps = Cabana::Grid::createReferenceConjugateGradient<double, MemorySpace>(
            *vector_layout );
        // The velocity corrector will create the relevant preconditioner here
        // since it depends on the matrix values
        return std::make_shared<CabanaFluids::VelocityCorrector<
            NumSpaceDims, ExecutionSpace, MemorySpace, reference_solver_type>>(
            pm, bc, ps, density, delta_t );
    }
    else
    {
        auto ps = Cabana::Grid::createHypreStructuredSolver<double, MemorySpace>(
            solver, *vector_layout );
        if ( precon.compare( "None" ) != 0 && precon.compare( "none" ) != 0 )
        {
            auto preconditioner =
                Cabana::Grid::createHypreStructuredSolver<double, MemorySpace>(
                    precon, *vector_layout, true );
            ps->setPreconditioner( preconditioner );
        }
        return std::make_shared<CabanaFluids::VelocityCorrector<
            NumSpaceDims, ExecutionSpace, MemorySpace, hypre_solver_type>>(
            pm, bc, ps, density, delta_t );
    }
}

} // namespace CabanaFluids

#endif // CABANAFLUIDS_VELOCITYCORRECTOR
