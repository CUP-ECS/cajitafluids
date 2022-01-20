/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Boundary Conditions for ExaCLAMR Shallow Water Solver
 */

#ifndef EXACLAMR_BOUNDARYCONDITIONS_HPP
#define EXACLAMR_BOUNDARYCONDITIONS_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Mesh.hpp>

#include <Kokkos_Core.hpp>

namespace CajitaFluids {
    /**
 * @struct BoundaryType
 * @brief Struct which contains enums of boundary type options for each boundary
 * In the current system, these are only used for setting up the matrix in the 
 * pressure solve.
 */
struct BoundaryType {
    enum Values {
        SOLID      = 0,
        FREE       = 1,
    };
};

/**
 * @struct BoundaryCondition
 * @brief Struct that applies the specified boundary conditions with a Kokkos Inline Function
 */
template <std::size_t NumSpaceDim> struct BoundaryCondition;

// Note that we loop over teh entire domain here so that later code, which might have geometry
// inside the domain, could use the same approach.
template <> struct BoundaryCondition<2> {
        template <class ArrayType>
        KOKKOS_INLINE_FUNCTION void operator()( const int gi, const int gj, 
						const int i, const int j, 
						ArrayType &matentry, 
                                                const double scale ) const {
	    // Our boundary conditions manifest mainly as updates to the pressure solve. Assume 
	    // the matrix has been initialized and we simply need to adjust the weights 
	    // for the boundaries.

            // Left Boundary
            if ( gi <= min[0] ) {
		matentry(i, j, 1) = 0; 
                if ( boundary_type[0] == BoundaryType::SOLID ) {
		    matentry(i, j, 0) -= scale;
		}
            }
            if ( gj <= min[1] ) {
		matentry(i, j, 3) = 0;
                if ( boundary_type[1] == BoundaryType::SOLID ) {
		    matentry(i, j, 0) -= scale;
		}
	    }
            if ( gi > max[0] - 1 ) {
		matentry(i, j, 2) = 0; 
                if ( boundary_type[2] == BoundaryType::SOLID ) {
		    matentry(i, j, 0) -= scale;
                }
            }
            if ( gj > max[1] - 1 ) {
		matentry(i, j, 4) = 0; 
                if ( boundary_type[3] == BoundaryType::SOLID ) {
		   matentry(i, j, 0) -= scale;
                }
            }
        }

        Kokkos::Array<int, 4> boundary_type; /**< Boundary condition type on all walls  */
        Kokkos::Array<int, 2> min;
        Kokkos::Array<int, 2> max;
    };

} // namespace CajitaFluids

#endif
