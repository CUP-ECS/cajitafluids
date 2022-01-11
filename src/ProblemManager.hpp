/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Problem manager class that stores the mesh and the state data and performs scatters and gathers
 */

#ifndef CABANAFLUIDS_PROBLEMMANAGER_HPP
#define CABANAFLUIDS_PROBLEMMANAGER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <FluidTypes.hpp>
#include <Mesh.hpp>


namespace CabanaFluids {

/**
 * @namespace Field
 * @brief Field namespace to track state array entities
 **/
namespace Field {
/**
 * @struct Momentum
 * @brief Momentum Field
 **/
struct Pressure {};

/**
 * @struct VelocityX
 * @brief Staggered Velocity Component on a face of the mesh
 **/
struct Velocity {};

}; // end namespace Field

**
 * The ProblemManager Class
 * @class ProblemManager
 * @brief ProblemManager class to store the mesh and state values, and 
 * to perform gathers and scatters in the approprate number of dimensions. To
 * start we just implement this in 2d. 
 **/
template <std::size_t NumSpaceDim, class MemorySpace>
class ProblemManager;


template <class MemorySpace>
class ProblemManager<2>;
{
  // Pull in a few Cajita type declarations we use a lot
  using Cajita::Cell;
  using Cajita::Face;
  using Cajita::Dim;

  public:
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    using iface_array = Cajita::Array<double, Face<Dim::I>,
                                      Cajita::UniformMesh<double>, MemorySpace>;
    using jface_array = Cajita::Array<double, Face<Dim::J>,
                                      Cajita::UniformMesh<double>, MemorySpace>;
    using cell_array = Cajita::Array<double, Cajita::Cell,
                                     Cajita::UniformMesh<double>, MemorySpace>;

    using halo = Cajita::Halo<MemorySpace>;
    using mesh_type = Mesh<MemorySpace>;

    template <class InitFunc, class ExecutionSpace>
    ProblemManager( const ExecutionSpace& exec_space,
                    const std::shared_ptr<mesh_type>& mesh,
                    const InitFunc& create_functor )
        : _mesh( mesh )
        // , other initializers
    {
        auto iface_scalar_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 1, 
                                       Face<Dim::I>() );
        auto jface_scalar_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 1, 
                                       Face<Dim::J>() );
        auto cell_scalar_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 1, Cajita::Cell() );

        _pressure = Cajita::createArray<double, MemorySpace>("pressure",
                                                             cell_scalar_layout);
        _ux_half = Cajita::createArray<double, MemorySpace>(
            "ux_half", iface_scalar_layout);
        _uy_half = Cajita::createArray<double, MemorySpace>(
            "uy_half", jface_scalar_layout);

        // Check what the right halo patterns should be here XXX
        _iface_halo = Cajita::createHalo<double, MemorySpace>(
            *iface_scalar_layout, Cajita::FaceHaloPattern<2>() );
        _jface_halo = Cajita::createHalo<double, MemorySpace>(
            *jface_scalar_layout, Cajita::FaceHaloPattern<2>() );
        _pressure_halo = Cajita::createHalo<double, MemorySpace>(
            *cell_scalar_layout, Cajita::FullHaloPattern() );
        
        // Initialize State Values ( Height, Momentum )
        initialize( create_functor );
    }

    /**
     * Initializes state values in the cells
     * @param create_functor Initialization function
     **/
    template <class InitFunctor>
    void initialize( const InitFunctor &create_functor ){
        // DEBUG: Trace State Initialization
        if ( _mesh->rank() == 0 && DEBUG ) std::cout << "Initializing Cell Fields\n";

        // Get Local Grid and Local Mesh
        auto local_grid = *( _mesh->localGrid() );
        auto local_mesh = Cajita::createLocalMesh<device_type>( local_grid );


        // Get State Arrays
        auto p = get( Cajita::Cell(), Field::Pressure(), 0 );
        auto ux = get( Face<Dim::I>(), Field::Velocity(), 0 );
        auto uy = get( Face<Dim::J>(), Field::Velocity(), 0 );

        // Loop Over All Owned Cells ( i, j, k )
        auto own_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
        Kokkos::parallel_for(
            "Initializing", Cajita::createExecutionPolicy( ghost_cells, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Initialize State Vectors
                double pressure;

                // Get Coordinates Associated with Indices ( i, j, k )
                int     coords[3] = { i, j, k };
                state_t x[3];

                local_mesh.coordinates( Cajita::Cell(), coords, x );

                // Initialization Function
                create_functor( Cajita::Cell, Field::Pressure, coords, x, pressure);
                // Assign Values to State Views
                p( i, j, k ) = pressure;
            } );
        };

        // Loop Over All Owned I-Faces ( i, j, k )
        auto own_faces = local_grid.indexSpace( 
            Cajita::Own(), Face<Dim::I>(), Cajita::Local() );
        Kokkos::parallel_for(
            "Initializing", Cajita::createExecutionPolicy( own_faces, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Initialize State Vectors
                double ux_half;

                // Get Coordinates Associated with Indices ( i, j, k )
                int     coords[3] = { i, j, k };
                state_t x[3];

                local_mesh.coordinates( Cajita::Cell(), coords, x );

                // Initialization Function
                create_functor( Face<Dim::I>, Field::Velocity, 
			        coords, x, ux_half);
                // Assign Values to State Views
                ux( i, j, k ) = ux_half;
            } );
        // Loop Over All Owned J-Faces ( i, j, k )
        own_faces = local_grid.indexSpace( 
            Cajita::Own(), Face<Dim::I>(), Cajita::Local() );
        Kokkos::parallel_for(
            "Initializing", Cajita::createExecutionPolicy( own_faces, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Initialize State Vectors
                double uy_half;

                // Get Coordinates Associated with Indices ( i, j, k )
                int     coords[3] = { i, j, k };
                state_t x[3];

                local_mesh.coordinates( Cajita::Cell(), coords, x );

                // Initialization Function
                create_functor( Face<Dim::J>, Field::Velocity, 
			        coords, x, uy_half);
                // Assign Values to State Views
                uy( i, j, k ) = uy_half;
            } );
        };
    };

    /**
     * Return mesh
     * @return Returns Mesh object
     **/
    const std::shared_ptr<Mesh<MemorySpace>> &mesh() const {
        return _mesh;
    };

    /**
     * Return Pressure Field
     * @param Location::Cell
     * @param Field::Height
     * @return Returns Height state array at cell centers
     **/
    typename cell_array::view_type get( Location::Cell, Field::Pressiure ) const {
        return _pressure->view();
    }

    /**
     * Return I Face Velocity 
     * @param Location::XFace
     * @param Field::Velocity
     * @return Returns Height state array at cell centers
     **/
    typename cell_array::view_type get( Location::XFace, Field::Velocity ) const {
        return _ux_half->view();
    }

    /**
     * Return Y Face Velocity 
     * @param Location::YFace
     * @param Field::Velocity
     * @return Returns Height state array at cell centers
     **/
    typename cell_array::view_type get( Location::YFace, Field::Velocity ) const {
        return _uy_half->view();
    }


    /**
     * Scatter State Data to Neighbors
     * @param Location::Cell
     **/
    void scatter( Cajta::Cell ) const {
        _cell_state_halo->scatter( ExecutionSpace(), *_pressure);
    };
    void scatter( Face<Dim::I> ) const {
        _cell_state_halo->scatter( ExecutionSpace(), *_ux_half);
    };
    void scatter( Face<Dim::J> ) const {
        _cell_state_halo->scatter( ExecutionSpace(), *_uy_half);
    };


    /**
     * Gather State Data from Neighbors
     * @param Location::Cell
     **/
    void gather( Cajita::Cell ) const {
        _cell_state_halo->gather( ExecutionSpace(), *_pressure );
    }
    void gather( Face<Dim::I> ) const {
        _cell_state_halo->gather( ExecutionSpace(), *_ux_half);
    };
    void gather( Face<Dim::J> ) const {
        _cell_state_halo->gather( ExecutionSpace(), *_uy_half);
    };

  private:
    std::shared_ptr<cell_array> _pressure;
    std::shared_ptr<iface_array> _ux_half;
    std::shared_ptr<jface_array> _uy_half;
    std::shared_ptr<mesh_type> _mesh; /**< Mesh object */
    std::shared_ptr<halo> _iface_halo;
    std::shared_ptr<halo> _jface_halo;
    std::shared_ptr<halo> _cell_halo;
};

} // namespace CajitaFluids

#endif
