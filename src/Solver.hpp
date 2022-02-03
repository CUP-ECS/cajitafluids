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

#include <Mesh.hpp>
#include <ProblemManager.hpp>
#include <BoundaryConditions.hpp>
#include <InflowSource.hpp>
#include <BodyForce.hpp>
#include <SiloWriter.hpp>
//#include <TimeIntegrator.hpp>

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

#include <mpi.h>

namespace CajitaFluids
{
//---------------------------------------------------------------------------//
class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void solve( const double t_final, const int write_freq ) = 0;
};

template <std::size_t NumSpaceDim, class MemorySpace, class ExecutionSpace>
class Solver;

//---------------------------------------------------------------------------//
template <class MemorySpace, class ExecutionSpace>
class Solver<2, MemorySpace, ExecutionSpace> : public SolverBase
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cajita::UniformMesh<double, 2>;
    using cell_array = Cajita::Array<double, Cajita::Cell, mesh_type, MemorySpace>;

#ifdef HYPRE
    using solver_type = Cajita::HypreStructuredSolver<double, Cajita::Cell, MemorySpace>;
#else
    using solver_type = Cajita::ReferenceConjugateGradient<double, Cajita::Cell, mesh_type, MemorySpace>;
#endif


    using Cell = Cajita::Cell;
    using FaceI = Cajita::Face<Cajita::Dim::I>;
    using FaceJ = Cajita::Face<Cajita::Dim::J>;

    template <class InitFunc>
    Solver( MPI_Comm comm, const Kokkos::Array<double, 4>& global_bounding_box,
            const std::array<int, 2>& global_num_cell,
            const Cajita::BlockPartitioner<2>& partitioner,
            const double density,
            const InitFunc& create_functor,
            const BoundaryCondition<2>& bc,
	    const InflowSource<2> &source,
            const BodyForce<2> &body,
            const double delta_t )
        : _halo_min( 1 ), _density(density), _bc(bc), 
	  _source(source), _body(body), _dt( delta_t)
    {
        _mesh = std::make_shared<Mesh<2, ExecutionSpace, MemorySpace>>(
            global_bounding_box, global_num_cell, partitioner,
            _halo_min, comm );
	
	_bc.min = _mesh->minDomainGlobalCellIndex();
        _bc.max = _mesh->maxDomainGlobalCellIndex();

        _pm = std::make_shared<ProblemManager<2, ExecutionSpace, MemorySpace>>(
            ExecutionSpace(), _mesh, create_functor );
        // Set up Silo for I/O
        _silo = std::make_shared<SiloWriter<2, ExecutionSpace, MemorySpace>>( _pm );

	auto vector_layout =
	    Cajita::createArrayLayout( _mesh->localGrid(), 1, Cell() );
	auto matrix_layout =
	    Cajita::createArrayLayout( _mesh->localGrid(), 5, Cell() );

	_lhs = Cajita::createArray<double, MemorySpace>("pressure LHS",
							vector_layout);
	_rhs = Cajita::createArray<double, MemorySpace>("pressure RHS",
							vector_layout);

#ifdef HYPRE
	// Create a solver and build the initial matrix - we use preconditioned
	// conjugate gradient by default.
	_pressure_solver = Cajita::createHypreStructuredSolver<double, 
            MemorySpace>( "PCG", *vector_layout );
       	auto preconditioner = Cajita::createHypreStructuredSolver<double,
            MemorySpace>( "Diagonal", *vector_layout, true );
        _pressure_solver->setPreconditioner( preconditioner );
#else
   	_pressure_solver =
            Cajita::createReferenceConjugateGradient<double, 
                MemorySpace>( *vector_layout );

#endif
	initializeSolverMatrix();

		   
	_pressure_solver->setTolerance ( 1.0e-9 );
	_pressure_solver->setMaxIter ( 2000 );
	_pressure_solver->setPrintLevel( 1 );
	// We could create a preconditioner here if we wanted, but we are lazy.
	_pressure_solver->setup();

    }

    void solve( const double t_final, const int write_freq ) override
    {
        int t = 0;
	double time = 0.0;
	int num_step;

	_silo->siloWrite( strdup( "Mesh" ), t, time, _dt );
	num_step = t_final / _dt;

	while ( (time < t_final) ) 
	{
	    if ( 0 == _mesh->rank() && 0 == t % write_freq )
		printf( "Step %d / %d at time = %f\n", t, num_step, time );

	    // 1. Handle inflow and body forces.
	    addExternalInputs();

#if 1
	    // 2. Adjust the velocity field to be divergence-free 
	    _buildRHS();
	    Cajita::ArrayOp::assign( *_lhs, 0.0, Cajita::Own());
	    _pressure_solver->solve( *_rhs, *_lhs );
	    _apply_pressure();
#endif

#if 0
	    // 3. Exchange velocity halos for advection
	    // 3.1 XXX Do a velocity halo for computing advection
	    _pm->gather( FaceI(), Field::Velocity() );
	    _pm->gather( FaceJ(), Field::Velocity() );
	    // 3.2 XXX Do a time step of advection
	    //TimeIntegrator::step( ExecutionSpace(), *_pm, _dt, _bc );
#endif

	    // 4. Output mesh state periodically
	    if ( 0 == t % write_freq ) {
		_silo->siloWrite( strdup( "Mesh" ), t, time, _dt );
	    }
	    time += _dt;
	    t++;
	}
    }


    void initializeSolverMatrix()
    {
	// Create a 5-point 2d laplacian stencil.
	std::vector<std::array<int, 2>> stencil = {
	    { 0, 0 }, { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
	_pressure_solver->setMatrixStencil( stencil );

	// Create the matrix entries. The stencil is defined over cells.
	auto local_grid = *( _mesh->localGrid() );

#ifdef HYPRE
	auto matrix_entry_layout = Cajita::createArrayLayout( _mesh->localGrid(), 5, Cell() );
	auto matrix_entries = Cajita::createArray<double, MemorySpace>(
	    "matrix_entries", matrix_entry_layout );
	auto entry_view = matrix_entries->view();
#else
        const auto& matrix_entries = _pressure_solver->getMatrixValues();
	auto entry_view = matrix_entries.view();
#endif

	// Build the solver matrix - set the default entry for each cell
	// then apply the boundary conditions to it
	auto owned_space = local_grid.indexSpace( Cajita::Own(), Cell(), Cajita::Local() );
	auto l2g = Cajita::IndexConversion::createL2G( *( _mesh->localGrid() ),
							Cell());
	auto scale = _dt / (_density * _mesh->cellSize() * _mesh->cellSize());
	Kokkos::parallel_for("fill_matrix_entries",
	    createExecutionPolicy( owned_space, ExecutionSpace() ),
	    KOKKOS_CLASS_LAMBDA( const int i, const int j ) {
		int gi, gj;
		l2g(i, j, gi, gj);
		entry_view( i, j, 0 ) = 4.0*scale;
		entry_view( i, j, 1 ) = -1.0*scale;
		entry_view( i, j, 2 ) = -1.0*scale;
		entry_view( i, j, 3 ) = -1.0*scale;
		entry_view( i, j, 4 ) = -1.0*scale;
		_bc.build_matrix(gi, gj, i, j, entry_view, scale);
	    });

#ifndef HYPRE
        std::vector<std::array<int, 2>> diag_stencil = { { 0, 0 } };
        _pressure_solver->setPreconditionerStencil( diag_stencil );
        const auto& preconditioner_entries = _pressure_solver->getPreconditionerValues();
        auto preconditioner_view = preconditioner_entries.view();
        Kokkos::parallel_for(
            "fill_preconditioner_entries",
            createExecutionPolicy( owned_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                preconditioner_view( i, j, 0 ) = 1.0 / (4.0 * scale);
            } );
#endif

#if DEBUG
        std::cout << "Matrix Rows:\n";
	std::cout << std::fixed;
	std::cout << std::showpoint;
        std::cout << std::setprecision(1);
        for (int i = 0; i < owned_space.extent(0); i++) {
            for (int j = 0; j < owned_space.extent(1); j++) {
		std::cout << "(" << i << "," << j << "): ";
		for (int k = 0; k < owned_space.size(); k++) {
		    int found = 0;
		    for (int s = 0; s < stencil.size(); s++) {
		        int iidx = i + owned_space.min(0);
		        int jidx = j + owned_space.min(1);
			int otheri = i + stencil[s][0],
			    otherj = j + stencil[s][1]; 
			if ((otheri < 0) || (otherj < 0) 
			    || (otheri >= owned_space.extent(0))
			    || (otherj >= owned_space.extent(1))) {
                            assert(entry_view(iidx, jidx, s) == 0);
			    continue;
			}
			int otheridx = otheri * owned_space.extent(0) + otherj;
			if (k == otheridx) {
		            int otheriidx = otheri + owned_space.min(0);
		            int otherjidx = otherj + owned_space.min(1);
			    std::cout << std::right << std::setw(5) << entry_view(iidx, jidx, s) << " ";
			    found = 1;
                        }
		    }
		    if (!found) std::cout << std::right << std::setw(5) << 0.0 << " ";
		}
                std::cout << "\n";
	    }
        }
#endif

#ifdef HYPRE
	_pressure_solver->setMatrixValues( *matrix_entries );
#endif
    }

    /* Internal methods for the solver */
    void addExternalInputs()
    {
        auto local_grid = *( _mesh->localGrid() );
        auto local_mesh = *( _mesh->localMesh() );
	double cell_size = _mesh->cellSize();
        double cell_area = cell_size * cell_size;

        auto owned_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

        auto quantity = _pm->get( Cell(), Field::Quantity() );
        Kokkos::parallel_for( "add external quantity",
            createExecutionPolicy( owned_cells, ExecutionSpace() ),
            KOKKOS_CLASS_LAMBDA( const int i, const int j ) {
		int idx[2] = {i, j};
		double loc[2];
	        local_mesh.coordinates( Cell(), idx, loc);
		double x = loc[0],
		       y = loc[1];
                _source(Cajita::Cell(), quantity, i, j, x, y, _dt, cell_area);
                _body(Cajita::Cell(), quantity, i, j, x, y, _dt, cell_area);
            });

        auto owned_ifaces = local_grid.indexSpace( Cajita::Own(), FaceI(), Cajita::Local() );
        auto ui  = _pm->get( FaceI(), Field::Velocity() );
        Kokkos::parallel_for( "add external x velocity",
            createExecutionPolicy( owned_ifaces, ExecutionSpace() ),
            KOKKOS_CLASS_LAMBDA( const int i, const int j ) {
		int idx[2] = {i, j};
		double loc[2];
	        local_mesh.coordinates( FaceI(), idx, loc);
		double x = loc[0],
		       y = loc[1];
                _source(FaceI(), ui, i, j, x, y, _dt, cell_area);
                _body(FaceI(), ui, i, j, x, y, _dt, cell_area);
            });

        auto owned_jfaces = local_grid.indexSpace( Cajita::Own(), FaceJ(), Cajita::Local() );
        auto uj  = _pm->get( FaceJ(), Field::Velocity() );
        Kokkos::parallel_for( "add external y velocity",
            createExecutionPolicy( owned_jfaces, ExecutionSpace() ),
            KOKKOS_CLASS_LAMBDA( const int i, const int j ) {
		int idx[2] = {i, j};
		double loc[2];
	        local_mesh.coordinates( FaceJ(), idx, loc);
		double x = loc[0],
		       y = loc[1];
                _source(FaceJ(), uj, i, j, x, y, _dt, cell_area);
                _body(FaceJ(), uj, i, j, x, y, _dt, cell_area);
            });
    }

    void _apply_pressure()
    {
        auto scale = _dt / (_density * _mesh->cellSize());
        auto u  = _pm->get( FaceI(), Field::Velocity() );
        auto v  = _pm->get( FaceJ(), Field::Velocity() );
	auto p  = _lhs->view();
        auto l2g = Cajita::IndexConversion::createL2G( *( _mesh->localGrid() ),
                                                        Cell());
        auto local_grid =  _mesh->localGrid();
        auto cell_space = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(),
                                                  Cajita::Local() );

	/* Now apply the LHS to adjust the velocity field. XXX Do we need to 
	 * halo the lhs here??? XXX */
        Kokkos::parallel_for(
            "apply pressure", createExecutionPolicy( cell_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                u(i, j, 0) -= scale * (p(i, j, 0) - p(i-1, j  , 0));
                v(i, j, 0) -= scale * (p(i, j, 0) - p(i,   j-1, 0));

		int gi, gj;
                l2g(i, j, gi, gj); 
		_bc.apply_pressure(gi, gj, i, j, u, v, scale);
            });
    }

    void _buildRHS() 
    {
        // Zero the RHS
	Cajita::ArrayOp::assign( *_rhs, 0.0, Cajita::Own());
        auto scale = 1.0 / _mesh->cellSize();

        auto u  = _pm->get( FaceI(), Field::Velocity() );
        auto v  = _pm->get( FaceJ(), Field::Velocity() );

        // For now we manually compute the divergence from the staggered
	// mesh for simplicity. Later we will want to use a G2G interface
        // in Cajita similar to its G2P interface, but that's not been 
        // developed yet. 
        auto local_grid = _mesh->localGrid();
        auto cell_space = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(),
                                                  Cajita::Local() );
        auto rhs = _rhs->view();

        // Snce we're not using the Cajita interpolication interface, we have
        // to halo manually
        _pm->gather( FaceI(), Field::Velocity() );
        _pm->gather( FaceJ(), Field::Velocity() );

        Kokkos::parallel_for(
            "divergence", createExecutionPolicy( cell_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                rhs(i, j, 0) = -scale * 
			(u(i+1, j, 0) - u(i, j, 0) 
			 + v(i, j+1, 0) - v(i, j, 0));
            });
#if DEBUG
        std::cout << "RHS:\n";
        auto owned_space = _mesh->localGrid()->indexSpace( Cajita::Own(), Cell(), Cajita::Local() );
        for (int i = owned_space.min(0); i < owned_space.max(0); i++) {
            for (int j = owned_space.min(1); j < owned_space.max(1); j++) {
		std::cout << "(" << (i - owned_space.min(0))
			  << "," << (j - owned_space.min(1)) << "): ";
		std::cout << rhs(i, j, 0) << "\n";
	    }
        }
	std::cout << std::flush;
#endif
    }

  private:

    /* Solver state variables */
    double _dt;
    double _density;
    double _cell_area;
    BodyForce<2> _body;
    InflowSource<2> _source;
    BoundaryCondition<2> _bc;
    int _halo_min;
    std::shared_ptr<Mesh<2, ExecutionSpace, MemorySpace>> _mesh;
    std::shared_ptr<ProblemManager<2, ExecutionSpace, MemorySpace>> _pm;
    std::shared_ptr<SiloWriter<2, ExecutionSpace, MemorySpace>> _silo;
    std::shared_ptr<cell_array> _lhs;
    std::shared_ptr<cell_array> _rhs;
    std::shared_ptr<solver_type> _pressure_solver;
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
              const double density,
              const InitFunc& create_functor,
              const BoundaryCondition<2>& bc,
              const InflowSource<2>& source,
              const BodyForce<2>& body,
              const double delta_t) 
{
    if ( 0 == device.compare( "serial" ) )
    {
// Hypre with CUDA support breaks support for the serial solver. We'll need
// to set it up to use a different solver in that case
#if defined(KOKKOS_ENABLE_SERIAL) && !defined(KOKKOS_ENABLE_CUDA)
        return std::make_shared<
            CajitaFluids::Solver<2, Kokkos::HostSpace, Kokkos::Serial>>(
            comm, global_bounding_box, global_num_cell, partitioner,
            density, create_functor, bc, source, body, delta_t);
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "openmp" ) )
    {
#if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA)
        return std::make_shared<
            CajitaFluids::Solver<2, Kokkos::HostSpace, Kokkos::OpenMP>>(
            comm, global_bounding_box, global_num_cell, partitioner,
            density, create_functor, bc, source, body, delta_t );
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "cuda" ) )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return std::make_shared<
            CajitaFluids::Solver<2, Kokkos::CudaSpace, Kokkos::Cuda>>(
            comm, global_bounding_box, global_num_cell, partitioner,
            density, create_functor, bc, source, body, delta_t );
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "hip" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<CajitaFluids::Solver<2, Kokkos::Experimental::HIPSpace,
                                               Kokkos::Experimental::HIP>>(
            comm, global_bounding_box, global_num_cell, partitioner,
            density, create_functor, bc, source, body, delta_t  );
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
