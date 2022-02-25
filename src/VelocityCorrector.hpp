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

#ifndef CAJITAFLUIDS_VELOCITYCORRECTOR_HPP
#define CAJITAFLUIDS_VELOCITYCORRECTOR_HPP

#include <Cajita_HypreStructuredSolver.hpp>
#include <Cajita_ReferenceStructuredSolver.hpp>
#include <Cajita_Types.hpp>

#include <Mesh.hpp>
#include <ProblemManager.hpp>
#include <BoundaryConditions.hpp>
#include <Interpolation.hpp>

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

#include <mpi.h>

namespace CajitaFluids
{
//---------------------------------------------------------------------------//
class VelocityCorrectorBase
{
  public:
    virtual ~VelocityCorrectorBase() = default;
    virtual void correctVelocity() = 0;
};

template <std::size_t NumSpaceDim, class ExecutionSpace, class MemorySpace, class SparseSolver>
class VelocityCorrector;

//---------------------------------------------------------------------------//

template <class ExecutionSpace, class MemorySpace, class SparseSolver>
class VelocityCorrector<2, ExecutionSpace, MemorySpace, SparseSolver> : public VelocityCorrectorBase
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cajita::UniformMesh<double, 2>;
    using cell_array = Cajita::Array<double, Cajita::Cell, mesh_type, MemorySpace>;
    using pm_type = ProblemManager<2, ExecutionSpace, MemorySpace>;
    using bc_type =  BoundaryCondition<2>;
    using hypre_solver_type = Cajita::HypreStructuredSolver<double, Cajita::Cell, MemorySpace>;
    using reference_solver_type = Cajita::ReferenceStructuredSolver<double, Cajita::Cell, mesh_type, MemorySpace>;
    using Cell = Cajita::Cell;
    using FaceI = Cajita::Face<Cajita::Dim::I>;

    using FaceJ = Cajita::Face<Cajita::Dim::J>;
  VelocityCorrector( const std::shared_ptr<pm_type> &pm,
		     bc_type &bc,
		     const std::shared_ptr<SparseSolver> &pressure_solver, 
		     double density,
		     double delta_t)
    : _pm(pm), _bc(bc), _density(density), _dt(delta_t), _pressure_solver(pressure_solver)
  {
    _mesh = pm->mesh();

    // Create the LHS and RHS vectors used calculate pressure correction amount 
    // at each cell
    auto vector_layout = Cajita::createArrayLayout( _mesh->localGrid(), 1, Cell() );
    _lhs = Cajita::createArray<double, MemorySpace>("pressure LHS",
						    vector_layout);
    _rhs = Cajita::createArray<double, MemorySpace>("pressure RHS",
						    vector_layout);
    Cajita::ArrayOp::assign(*_lhs, 0.0, Cajita::Ghost());
    Cajita::ArrayOp::assign(*_rhs, 0.0, Cajita::Ghost());

    // Set up the solver to compute the pressure at each point using a 5-point 
    // 2d laplacian stencil. The matrix itself will have 0 weights for elements
    // that fall outside the boundary, set up by the boundary condition object. 
    std::vector<std::array<int, 2>> stencil = {{ 0, 0 }, { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
    _pressure_solver->setMatrixStencil(stencil, false );

    // Fill the associated matrix assocuated with with values
    fillMatrixValues(_pressure_solver);

    _pressure_solver->setTolerance ( 1.0e-6 );
    _pressure_solver->setMaxIter ( 2000 );
    _pressure_solver->setPrintLevel( 1 );
    _pressure_solver->setup();

    // Finally, we need to halo pressure values with neighbors with whom
    // we share a face so that we can correct velocities on those faces. Note
    // that this is a much simpler and shallower halo poattern than the ones
    // used for advection.
    _pressure_halo = Cajita::createHalo<double, MemorySpace>(
                         *vector_layout, Cajita::FaceHaloPattern<2>(), 1);
  }

  template <class View_t> 
  void initializeMatrixValues(const View_t &entry_view)
  {
    // Create the matrix entries. The stencil is defined over cells.
    auto local_grid = _mesh->localGrid();
    
    // Build the solver matrix - set the default entry for each cell
    // then apply the boundary conditions to it
    auto owned_space = local_grid->indexSpace( Cajita::Own(), Cell(), Cajita::Local() );
    auto l2g = Cajita::IndexConversion::createL2G( *( _mesh->localGrid() ),
						   Cell());
    auto scale = _dt / (_density * _mesh->cellSize() * _mesh->cellSize());
    const bc_type &bc = _bc;

    Kokkos::parallel_for("fill_matrix_entries",
			 createExecutionPolicy( owned_space, ExecutionSpace() ),
			 KOKKOS_LAMBDA( const int i, const int j ) {
			   int gi, gj;
			   l2g(i, j, gi, gj);
			   entry_view( i, j, 0 ) = 4.0*scale;
			   entry_view( i, j, 1 ) = -1.0*scale;
			   entry_view( i, j, 2 ) = -1.0*scale;
			   entry_view( i, j, 3 ) = -1.0*scale;
			   entry_view( i, j, 4 ) = -1.0*scale;
			   bc.build_matrix(gi, gj, i, j, entry_view, scale);
			 });
  }

  // Specialization of matrix fill for Hypre
  void fillMatrixValues(std::shared_ptr<hypre_solver_type> &solver)
  {
    auto matrix_entry_layout = Cajita::createArrayLayout( _mesh->localGrid(), 5, Cell() );
    auto matrix_entries = Cajita::createArray<double, MemorySpace>("matrix_entries", matrix_entry_layout );
    initializeMatrixValues(matrix_entries->view());
    solver->setMatrixValues( *matrix_entries );
  }

  // Specialization of matrix fill for the Cajita structured solver
  void fillMatrixValues(std::shared_ptr<reference_solver_type> &solver)
  {
    const auto& matrix_entries = solver->getMatrixValues();
    auto m = matrix_entries.view();
    initializeMatrixValues(m);

    // Go ahead and set up a simple diagonal preconditioner for the reference solver now
    std::vector<std::array<int, 2>> diag_stencil = { { 0, 0 } };
    solver->setPreconditionerStencil( diag_stencil, false );
    const auto& preconditioner_entries = solver->getPreconditionerValues();
    auto local_grid = _mesh->localGrid();
    auto owned_space = local_grid->indexSpace( Cajita::Own(), Cell(), Cajita::Local() );
    auto preconditioner_view = preconditioner_entries.view();
    auto scale = _dt / (_density * _mesh->cellSize() * _mesh->cellSize());

    Kokkos::parallel_for(
			 "fill_preconditioner_entries",
			 createExecutionPolicy( owned_space, ExecutionSpace() ),
			 KOKKOS_LAMBDA( const int i, const int j ) {
			   preconditioner_view( i, j, 0 ) = 1.0 / m(i, j, 0);
			 } );
  }

  template <class View_t> 
  void printMatrixEntries( View_t entry_view )
  {
    auto local_grid = ( _mesh->localGrid() );
    std::vector<std::array<int, 2>> stencil = {{ 0, 0 }, { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };

    // Build the solver matrix - set the default entry for each cell
    // then apply the boundary conditions to it
    auto owned_space = local_grid->indexSpace( Cajita::Own(), Cell(), Cajita::Local() );
    
    auto entry_view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), entry_view);
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
	      assert(entry_view_host(iidx, jidx, s) == 0);
	      continue;
	    }
	    int otheridx = otheri * owned_space.extent(0) + otherj;
	    if (k == otheridx) {
	      int otheriidx = otheri + owned_space.min(0);
	      int otherjidx = otherj + owned_space.min(1);
	      std::cout << std::right << std::setw(5) << entry_view_host(iidx, jidx, s) << " ";
	      found = 1;
	    }
	  }
	  if (!found) std::cout << std::right << std::setw(5) << 0.0 << " ";
	}
	std::cout << "\n";
      }
    }
  }

    void _buildRHS() 
    {
        // Zero the RHS
	Cajita::ArrayOp::assign( *_rhs, 0.0, Cajita::Own());
        auto scale = 1.0 / _mesh->cellSize();

        auto u  = _pm->get( FaceI(), Field::Velocity(), Version::Current() );
        auto v  = _pm->get( FaceJ(), Field::Velocity(), Version::Current() );

        // For now we manually compute the divergence from the staggered
	// mesh for simplicity. Later we will want to use a G2G interface
        // in Cajita similar to its G2P interface, but that's not been 
        // developed yet. 
        auto local_grid = _mesh->localGrid();
        auto cell_space = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(),
                                                  Cajita::Local() );
        auto rhs = _rhs->view();

	// Get the ghosts we'll need for interpolation XXX do we need this?
        _pm->gather( Version::Current() );

        Kokkos::parallel_for(
            "divergence", createExecutionPolicy( cell_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
	        rhs(i, j, 0) = -scale * (u(i+1, j, 0) - u(i, j, 0) 
					  + v(i, j+1, 0) - v(i, j, 0));
            });
#if DEBUG
	auto rhs_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs);
        std::cout << "RHS:\n";
        auto owned_space = _mesh->localGrid()->indexSpace( Cajita::Own(), Cell(), Cajita::Local() );
        for (int i = owned_space.min(0); i < owned_space.max(0); i++) {
            for (int j = owned_space.min(1); j < owned_space.max(1); j++) {
		std::cout << "(" << (i - owned_space.min(0))
			  << "," << (j - owned_space.min(1)) << "): ";
		std::cout << rhs_host(i, j, 0) << "\n";
	    }
        }
	std::cout << std::flush;
#endif
    }

    void _apply_pressure()
    {
        auto scale = _dt / (_density * _mesh->cellSize());

        // Modifies the current velocity field *in place* to be divergence free
        auto u  = _pm->get( FaceI(), Field::Velocity(), Version::Current() );
        auto v  = _pm->get( FaceJ(), Field::Velocity(), Version::Current() );
	auto p  = _lhs->view();

        auto l2g = Cajita::IndexConversion::createL2G( *( _mesh->localGrid() ),
                                                        Cell());
        auto local_grid =  _mesh->localGrid();
        auto cell_space = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(),
                                                  Cajita::Local() );

	/* Now apply the LHS to adjust the velocity field. We need to
	 * halo the lhs here to adjut edge velocities. This halo is
	 * strictly larger than needed (only needs to be 1 deep and
	 * not include corners), but we reuse the exiting halo
	 * pattern for simplicity. */
	_pressure_halo->gather( ExecutionSpace(), *_lhs);

	const bc_type &bc = _bc;

        Kokkos::parallel_for(
            "apply pressure", createExecutionPolicy( cell_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                u(i, j, 0) -= scale * (p(i, j, 0) - p(i-1, j  , 0));
                v(i, j, 0) -= scale * (p(i, j, 0) - p(i,   j-1, 0));

		int gi, gj;
                l2g(i, j, gi, gj); 
		bc(Cell(), u, v, gi, gj, i, j);
            });
    }

  void correctVelocity()
  {
    _buildRHS();
    Cajita::ArrayOp::assign( *_lhs, 0.0, Cajita::Own());
    _pressure_solver->solve( *_rhs, *_lhs );
    _apply_pressure();

  }
    private:
  std::shared_ptr<cell_array> _lhs;
  std::shared_ptr<cell_array> _rhs;
  BoundaryCondition<2> _bc;
  std::shared_ptr<pm_type> _pm;
  std::shared_ptr<Mesh<2, ExecutionSpace, MemorySpace>> _mesh;
  std::shared_ptr<SparseSolver> _pressure_solver;
  std::shared_ptr<Cajita::Halo<MemorySpace>> _pressure_halo;
  
  double _dt;
  double _density;
};

template <std::size_t NumSpaceDims, class ExecutionSpace, class MemorySpace,
	  class ProblemManagerType, class BoundaryConditionType>
std::shared_ptr<VelocityCorrectorBase>
createVelocityCorrector( const std::shared_ptr<ProblemManagerType> &pm,
			 BoundaryConditionType &bc,
			 const double density,
			 const double delta_t, 
			 std::string solver, 
			 std::string precon) 
{
    using mesh_type = Cajita::UniformMesh<double, 2>;
    using hypre_solver_type = Cajita::HypreStructuredSolver<double, Cajita::Cell, MemorySpace>;
    using reference_solver_type = Cajita::ReferenceStructuredSolver<double, Cajita::Cell, mesh_type, MemorySpace>;

      auto vector_layout =
	Cajita::createArrayLayout( pm->mesh()->localGrid(), 1, Cajita::Cell() );
    if (solver.compare("Reference") == 0) {
      auto ps = Cajita::createReferenceConjugateGradient<double, 
							 MemorySpace>( *vector_layout );
      // The velocity corrector will create the relevant preconditioner here 
      // since it depends on the matrix values
      return std::make_shared<CajitaFluids::VelocityCorrector<NumSpaceDims, ExecutionSpace, MemorySpace, reference_solver_type>>(pm, bc, ps, density, delta_t);

      
    } else { 
      auto ps = Cajita::createHypreStructuredSolver<double, 
						    MemorySpace>( solver, *vector_layout );
      if (precon.compare("None") != 0 && precon.compare("none") != 0) {
          auto preconditioner = Cajita::createHypreStructuredSolver<double,
								MemorySpace>( precon, *vector_layout, true );
          ps->setPreconditioner(preconditioner);
      }
      return std::make_shared<CajitaFluids::VelocityCorrector<NumSpaceDims, ExecutionSpace, MemorySpace, hypre_solver_type>>(pm, bc, ps, density, delta_t);
    }
}

} // namespace CajitaFluids

#endif // CAJITAFLUIDS_VELOCITYCORRECTOR
