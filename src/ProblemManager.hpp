/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Problem manager class that stores the mesh and the state data and performs scatters and gathers
 */

#ifndef CAJITAFLUIDS_PROBLEMMANAGER_HPP
#define CAJITAFLUIDS_PROBLEMMANAGER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <Mesh.hpp>

namespace CajitaFluids {

/**
 * @namespace Field
 * @brief Field namespace to track state array entities
 **/
namespace Field {

/**
 * @struct Density
 * @brief Tag structure for the density of the advected quantity (which doesn't effect pressure!)
 **/
struct Density {};

/**
 * @struct Velocity
 * @brief Tag structure for the staggered velocity component on a face of the mesh
 **/
struct Velocity {};

}; // end namespace Field

/**
 * The ProblemManager Class
 * @class ProblemManager
 * @brief ProblemManager class to store the mesh and state values, and 
 * to perform gathers and scatters in the approprate number of dimensions.
 **/
template <std::size_t NumSpaceDim, class ExecutionSpace, class MemorySpace>
class ProblemManager;

using Cajita::Dim;

/* The 2D implementation of hte problem manager class */
template <class ExecutionSpace, class MemorySpace>
class ProblemManager<2, ExecutionSpace, MemorySpace>
{
  // Pull in a few Cajita type declarations we use a lot

  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using iface_array = Cajita::Array<double, Cajita::Face<Cajita::Dim::I>,
                                      Cajita::UniformMesh<double, 2>, MemorySpace>;
    using jface_array = Cajita::Array<double, Cajita::Face<Cajita::Dim::J>,
                                      Cajita::UniformMesh<double, 2>, MemorySpace>;
    using cell_array = Cajita::Array<double, Cajita::Cell,
                                     Cajita::UniformMesh<double, 2>, MemorySpace>;

    using halo = Cajita::Halo<MemorySpace>;
    using mesh_type = Mesh<2, MemorySpace>;

    template <class InitFunc>
    ProblemManager( const ExecutionSpace& exec_space,
                    const std::shared_ptr<mesh_type>& mesh,
                    const InitFunc& create_functor )
        : _mesh( mesh )
        // , other initializers
    {
        auto iface_scalar_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 1, 
                                       Cajita::Face<Cajita::Dim::I>() );
        auto jface_scalar_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 1, 
                                       Cajita::Face<Cajita::Dim::J>() );
        auto cell_scalar_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 1, Cajita::Cell() );

        // The density of the scalar quantity being advected
        _density = Cajita::createArray<double, MemorySpace>("density",
                                                            cell_scalar_layout);
        _ui = Cajita::createArray<double, MemorySpace>(
            "ui", iface_scalar_layout);
        _uj = Cajita::createArray<double, MemorySpace>(
            "uj", jface_scalar_layout);

        // Check what the right halo patterns should be here XXX
        _iface_halo = Cajita::createHalo<double, MemorySpace>(
            *iface_scalar_layout, Cajita::FaceHaloPattern<2>() );
        _jface_halo = Cajita::createHalo<double, MemorySpace>(
            *jface_scalar_layout, Cajita::FaceHaloPattern<2>() );
        _cell_halo = Cajita::createHalo<double, MemorySpace>(
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
        auto ui = get( Cajita::Face<Cajita::Dim::I>(), Field::Velocity(), 0 );
        auto uj = get( Cajita::Face<Cajita::Dim::J>(), Field::Velocity(), 0 );

        // Loop Over All Owned Cells ( i, j, k )
        auto own_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
        Kokkos::parallel_for(
            "Initializing", Cajita::createExecutionPolicy( own_cells, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Get Coordinates Associated with Indices ( i, j, k )
                int     coords[3] = { i, j, k };
                double x[3];

                local_mesh.coordinates( Cajita::Cell(), coords, x );

                // Initialization Function
                create_functor( Cajita::Cell(), Field::Pressure(), coords, x, p( i, j, k ));
            } );

        // Loop Over All Owned I-Faces ( i, j, k )
        auto own_faces = local_grid.indexSpace( 
            Cajita::Own(), Cajita::Face<Cajita::Dim::I>(), Cajita::Local() );
        Kokkos::parallel_for(
            "Initializing", Cajita::createExecutionPolicy( own_faces, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Get Coordinates Associated with Indices ( i, j, k )
                int     coords[3] = { i, j, k };
                double x[3];

                local_mesh.coordinates( Cajita::Cell(), coords, x );

                // Initialization Function
                create_functor( Cajita::Face<Cajita::Dim::I>(), Field::Velocity(), 
			        coords, x, ui( i, j, k));
            } );

        // Loop Over All Owned J-Faces ( i, j, k )
        own_faces = local_grid.indexSpace( 
            Cajita::Own(), Cajita::Face<Cajita::Dim::I>(), Cajita::Local() );
        Kokkos::parallel_for(
            "Initializing", Cajita::createExecutionPolicy( own_faces, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Get Coordinates Associated with Indices ( i, j, k )
                int     coords[3] = { i, j, k };
                double x[3];

                local_mesh.coordinates( Cajita::Cell(), coords, x );

                // Initialization Function
                create_functor( Cajita::Face<Cajita::Dim::J>(), Field::Velocity(), 
			        coords, x, uj( i, j, k) );
            } );
    }

    /**
     * Return mesh
     * @return Returns Mesh object
     **/
    const std::shared_ptr<Mesh<2, MemorySpace>> &mesh() const {
        return _mesh;
    };

    /**
     * Return Pressure Field
     * @param Location::Cell
     * @param Field::Height
     * @return Returns Height state array at cell centers
     **/
    typename cell_array::view_type get( Cajita::Cell, Field::Density ) const {
        return _density->view();
    }

    /**
     * Return I Face Velocity 
     * @param Location::XFace
     * @param Field::Velocity
     * @return Returns Height state array at cell centers
     **/
    typename cell_array::view_type get( Cajita::Face<Cajita::Dim::I>, Field::Velocity ) const {
        return _ui->view();
    }

    /**
     * Return Y Face Velocity 
     * @param Location::YFace
     * @param Field::Velocity
     * @return Returns Height state array at cell centers
     **/
    typename cell_array::view_type get( Cajita::Face<Cajita::Dim::J>, Field::Velocity ) const {
        return _uj->view();
    }


    /**
     * Scatter State Data to Neighbors
     * @param Location::Cell
     **/
    void scatter( Cajita::Cell ) const {
        _cell_halo->scatter( ExecutionSpace(), *_density);
    };
    void scatter( Cajita::Face<Cajita::Dim::I> ) const {
        _iface_halo->scatter( ExecutionSpace(), *_ui);
    };
    void scatter( Cajita::Face<Cajita::Dim::J> ) const {
        _jface_halo->scatter( ExecutionSpace(), *_uj);
    };


    /**
     * Gather State Data from Neighbors
     * @param Location::Cell
     **/
    void gather( Cajita::Cell ) const {
        _cell_halo->gather( ExecutionSpace(), *_density );
    }
    void gather( Cajita::Face<Cajita::Dim::I> ) const {
        _iface_halo->gather( ExecutionSpace(), *_ui);
    };
    void gather( Cajita::Face<Cajita::Dim::J> ) const {
        _jface_halo->gather( ExecutionSpace(), *_uj);
    };

  private:
    std::shared_ptr<cell_array> _density;
    std::shared_ptr<iface_array> _ui;
    std::shared_ptr<jface_array> _uj;
    std::shared_ptr<mesh_type> _mesh; /**< Mesh object */
    std::shared_ptr<halo> _iface_halo;
    std::shared_ptr<halo> _jface_halo;
    std::shared_ptr<halo> _cell_halo;
};

} // namespace CajitaFluids

#endif // CAJITAFLUIDS_PROBLEMMANAGER_HPP
