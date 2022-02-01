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
 * @struct Quantity
 * @brief Tag structure for the quantity of the advected quantity (which doesn't effect pressure!)
 **/
struct Quantity {};

/**
 * @struct Velocity
 * @brief Tag structure for the magnitude of the normal velocity component on 
 * faces of the mesh
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


/* The 2D implementation of hte problem manager class */
template <class ExecutionSpace, class MemorySpace>
class ProblemManager<2, ExecutionSpace, MemorySpace>
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using cell_array = Cajita::Array<double, Cajita::Cell,
                                     Cajita::UniformMesh<double, 2>, MemorySpace>;
    using iface_array = Cajita::Array<double, Cajita::Face<Cajita::Dim::I>,
                                      Cajita::UniformMesh<double, 2>, MemorySpace>;
    using jface_array = Cajita::Array<double, Cajita::Face<Cajita::Dim::J>,
                                      Cajita::UniformMesh<double, 2>, MemorySpace>;

    // Meaningless type for now until we have 3D support in.
    using kface_array = Cajita::Array<double, Cajita::Face<Cajita::Dim::K>,
                                      Cajita::UniformMesh<double, 2>, MemorySpace>;
    using halo = Cajita::Halo<MemorySpace>;

    using mesh_type = Mesh<2, ExecutionSpace, MemorySpace>;

    template <class InitFunc>
    ProblemManager( const ExecutionSpace& exec_space,
                    const std::shared_ptr<mesh_type>& mesh,
                    const InitFunc& create_functor )
        : _mesh( mesh )
        // , other initializers
    {
	// The layouts of our various arrays for values on the staggered mesh and 
	// other associated data strutures. Do these needto be version with halos 
	// associuated with them? XXX
        auto iface_scalar_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 1, 
                                       Cajita::Face<Cajita::Dim::I>() );
        auto jface_scalar_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 1, 
                                       Cajita::Face<Cajita::Dim::J>() );
        auto cell_scalar_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 1, Cajita::Cell() );

	// Our halo for velocities and cell population information is
	// 2 deep - semi-lagrangian advection moves data between cells
        // no farther than 2 steps, and our timestep should be much smaller than
        // this.

	// The actual arrays storing mesh quantities
        // 1. The quantity of the scalar quantity being advected
        _quantity = Cajita::createArray<double, MemorySpace>("quantity",
                                                            cell_scalar_layout);
	
	// 2. The magnitudes of the velocities normal to the cell faces
        _ui = Cajita::createArray<double, MemorySpace>( "ui", iface_scalar_layout);
        _uj = Cajita::createArray<double, MemorySpace>( "uj", jface_scalar_layout);

        // Halo patterns for the velocity and quantity halos. These halos are 
	// used for advection calculations, and are two cells deep as that is the
	// the maximum that we allow quantities to advect in a single timestep.
        auto halo_pattern = Cajita::HaloPattern<2>();
        std::vector<std::array<int, 2>> neighbors;
        for ( int i = -2; i < 3; i++ ) {
            for ( int j = -2; j < 3; j++ ) {
                if (  !( i == 0 && j == 0 ) ) {
                    neighbors.push_back( { i, j } );
                }
            }
        }

        _iface_halo = Cajita::createHalo<double, MemorySpace>(
            *iface_scalar_layout, halo_pattern );
        _jface_halo = Cajita::createHalo<double, MemorySpace>(
            *jface_scalar_layout, halo_pattern );
        _cell_halo = Cajita::createHalo<double, MemorySpace>(
            *cell_scalar_layout, halo_pattern );
   
        // Initialize State Values ( quantity and velocity )
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
        auto local_mesh = *( _mesh->localMesh() );
	double cell_size = _mesh->cellSize();

        // Get State Arrays
        auto q = get( Cajita::Cell(), Field::Quantity() );

        // Loop Over All Owned Cells ( i, j )
        auto own_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
        int index[2] = {0, 0};
        double loc[2]; // x/y loocation of the cell at 0, 0
	local_mesh.coordinates( Cajita::Cell(), index, loc); 
        Kokkos::parallel_for(
            "Initialize Cells`", Cajita::createExecutionPolicy( own_cells, ExecutionSpace() ), 
            KOKKOS_LAMBDA( const int i, const int j ) {
                // Get Coordinates Associated with Indices ( i, j, k )
                int  coords[2] = { i, j };
                double x[2];
		x[0] = loc[0] + cell_size * i;
		x[1] = loc[1] + cell_size * j;
                // Initialization Function
                create_functor( Cajita::Cell(), Field::Quantity(), coords, x, 
                                q(i, j, 0));
            } );

        // Loop Over All Owned I-Faces ( i, j )
        auto own_faces = local_grid.indexSpace( 
            Cajita::Own(), Cajita::Face<Cajita::Dim::I>(), Cajita::Local() );
        auto ui = get( Cajita::Face<Cajita::Dim::I>(), Field::Velocity() );
	local_mesh.coordinates( Cajita::Face<Cajita::Dim::I>(), index, loc ); 
        Kokkos::parallel_for(
            "Initialize I-Faces", Cajita::createExecutionPolicy( own_faces, ExecutionSpace() ), 
            KOKKOS_LAMBDA( const int i, const int j ) {
                // Get Coordinates Associated with Indices ( i, j )
                int     coords[2] = { i, j };
                double x[2];
		x[0] = loc[0] + cell_size * i;
		x[1] = loc[1] + cell_size * j;

                // Initialization Function
                create_functor( Cajita::Face<Cajita::Dim::I>(), Field::Velocity(), 
			        coords, x, ui(i, j, 0));
            } );

        // Loop Over All Owned J-Faces ( i, j )
        own_faces = local_grid.indexSpace( 
            Cajita::Own(), Cajita::Face<Cajita::Dim::J>(), Cajita::Local() );
        auto uj = get( Cajita::Face<Cajita::Dim::J>(), Field::Velocity() );
	local_mesh.coordinates( Cajita::Face<Cajita::Dim::J>(), index, loc ); 
        Kokkos::parallel_for(
            "Initialize J-Faces", Cajita::createExecutionPolicy( own_faces, ExecutionSpace() ), 
            KOKKOS_LAMBDA( const int i, const int j ) {
                // Get Coordinates Associated with Indices ( i, j )
                int     coords[2] = { i, j };
                double x[2];

		x[0] = loc[0] + cell_size * i;
		x[1] = loc[1] + cell_size * j;

                // Initialization Function
                create_functor( Cajita::Face<Cajita::Dim::J>(), Field::Velocity(), 
			        coords, x, uj(i, j, 0) );
            } );
    };

    /**
     * Return mesh
     * @return Returns Mesh object
     **/
    const std::shared_ptr<Mesh<2, ExecutionSpace, MemorySpace>> &mesh() const {
        return _mesh;
    };

    /**
     * Return Quantity Field
     * @param Location::Cell
     * @param Field::Quantity
     * @return Returns view of advected quantity at cell centers
     **/
    typename cell_array::view_type get( Cajita::Cell, Field::Quantity ) const {
        return _quantity->view();
    };

    /**
     * Return I Face Velocity 
     * @param Face<Dim::I>
     * @param Field::Velocity
     * @return Returns view of norm velocity magnitude on i faces
     **/
    typename cell_array::view_type get( Cajita::Face<Cajita::Dim::I>, Field::Velocity ) const {
        return _ui->view();
    };

    /**
     * Return J Face Velocity 
     * @param Face<Dim::J>
     * @param Field::Velocity
     * @return Returns view of norm velocity magnitude on j faces
     **/
    typename cell_array::view_type get( Cajita::Face<Cajita::Dim::J>, Field::Velocity ) const {
        return _uj->view();
    };


    /**
     * Scatter State Data to Neighbors
     * @param Location::Cell
     **/
    void scatter( Cajita::Cell, Field::Quantity) const {
        _cell_halo->scatter( ExecutionSpace(), *_quantity);
    };

    /**
     * Scatter State Data to Neighbors
     * @param Location::Face<Dim::I>
     **/
    void scatter( Cajita::Face<Cajita::Dim::I>, Field::Velocity ) const {
        _iface_halo->scatter( ExecutionSpace(), *_ui);
    };

    /**
     * Scatter State Data to Neighbors
     * @param Location::Face<Dim::J>
     **/
    void scatter( Cajita::Face<Cajita::Dim::J>, Field::Velocity ) const {
        _jface_halo->scatter( ExecutionSpace(), *_uj);
    };


    /**
     * Gather State Data from Neighbors
     * @param Location::Cell
     **/
    void gather( Cajita::Cell, Field::Quantity ) const {
        _cell_halo->gather( ExecutionSpace(), *_quantity );
    };
    void gather( Cajita::Face<Cajita::Dim::I>, Field::Velocity ) const {
        _iface_halo->gather( ExecutionSpace(), *_ui);
    };
    void gather( Cajita::Face<Cajita::Dim::J>, Field::Velocity ) const {
        _jface_halo->gather( ExecutionSpace(), *_uj);
    };


  private:
    std::shared_ptr<cell_array> _quantity;
    std::shared_ptr<iface_array> _ui;
    std::shared_ptr<jface_array> _uj;
    std::shared_ptr<mesh_type> _mesh; /**< Mesh object */
    std::shared_ptr<halo> _iface_halo;
    std::shared_ptr<halo> _jface_halo;
    std::shared_ptr<halo> _cell_halo;
};

} // namespace CajitaFluids

#endif // CAJITAFLUIDS_PROBLEMMANAGER_HPP
