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
 * @brief Version namespace to track whether the current or next array version is requested
 **/
namespace Version {

/**
 * @struct Current
 * @brief Tag structure for the current values of field variables. Used when 
 * values are only being read or the algorithm allows the variable to be modified
 * in place
 **/
struct Current {};

/**
 * @struct Next
 * @brief Tag structure for the values of field variables at the next timestep. 
 * Used when values being written cannot be modified in place. Note that next
 * values are only written, current values are read or written.
 **/
struct Next {};

} // namespace Version

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
  
    using Cell = Cajita::Cell;
    using FaceI = Cajita::Face<Cajita::Dim::I>;
    using FaceJ = Cajita::Face<Cajita::Dim::J>;
    using FaceK = Cajita::Face<Cajita::Dim::K>;
  
    using cell_array = Cajita::Array<double, Cajita::Cell,
                                     Cajita::UniformMesh<double, 2>, MemorySpace>;
    using iface_array = Cajita::Array<double, Cajita::Face<Cajita::Dim::I>,
                                      Cajita::UniformMesh<double, 2>, MemorySpace>;
    using jface_array = Cajita::Array<double, Cajita::Face<Cajita::Dim::J>,
                                      Cajita::UniformMesh<double, 2>, MemorySpace>;

    // Meaningless type for now until we have 3D support in.
    using kface_array = Cajita::Array<double, Cajita::Face<Cajita::Dim::K>,
                                      Cajita::UniformMesh<double, 2>, MemorySpace>;
    using halo_type = Cajita::Halo<MemorySpace>;
    using mesh_type = Mesh<2, ExecutionSpace, MemorySpace>;

    template <class InitFunc>
    ProblemManager( const ExecutionSpace& exec_space,
                    const std::shared_ptr<mesh_type>& mesh,
                    const InitFunc& create_functor )
        : _mesh( mesh )
        // , other initializers
    {
	// The layouts of our various arrays for values on the staggered mesh
        // and other associated data strutures. Do there need to be version with
        // halos associuated with them? 
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
        _quantity_curr = Cajita::createArray<double, MemorySpace>("quantity",
                                                                  cell_scalar_layout);
        _quantity_next = Cajita::createArray<double, MemorySpace>("quantity",
                                                                  cell_scalar_layout);
	
	// 2. The magnitudes of the velocities normal to the cell faces
        _ui_curr = Cajita::createArray<double, MemorySpace>( "u0", iface_scalar_layout);
        _ui_next = Cajita::createArray<double, MemorySpace>( "u1", iface_scalar_layout);
        _uj_curr = Cajita::createArray<double, MemorySpace>( "v0", jface_scalar_layout);
        _uj_next = Cajita::createArray<double, MemorySpace>( "v1", jface_scalar_layout);

        // Halo patterns for the velocity and quantity advection. These halos are 
	// used for advection calculations, and are three cells deep because:
	// We only allow quantities to advect at most one cell in a single timestep,
	// but that cell could be outside our owned space (1 halo cell) and 
	// we interpolate the amount advected with a 3rd-order polynomial which 
	// could reach two additional cells in outside our boundary. As a result, we need 
	// halos that are 3 cells deep. It's unlikely that we need this deep at corner, but
	// we halo that far anyway, just to be sure.
        auto advection_halo_pattern = Cajita::HaloPattern<2>();
        std::vector<std::array<int, 2>> neighbors;
        for ( int i = -3; i <= 3; i++ ) {
            for ( int j = -3; j <= 3; j++ ) {
                if (  !( i == 0 && j == 0 ) ) {
                    neighbors.push_back( { i, j } );
                }
            }
        }
        advection_halo_pattern.setNeighbors( neighbors );

        _iface_advection_halo = Cajita::createHalo<double, MemorySpace>(
            *iface_scalar_layout, advection_halo_pattern );
        _jface_advection_halo = Cajita::createHalo<double, MemorySpace>(
            *jface_scalar_layout, advection_halo_pattern );
        _cell_advection_halo = Cajita::createHalo<double, MemorySpace>(
            *cell_scalar_layout, advection_halo_pattern );

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
        auto q = get( Cajita::Cell(), Field::Quantity(), Version::Current() );

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
        auto ui = get( Cajita::Face<Cajita::Dim::I>(), Field::Velocity(), Version::Current() );
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
        auto uj = get( Cajita::Face<Cajita::Dim::J>(), Field::Velocity(), Version::Current() );
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
     * @param Version::Current
     * @return Returns view of current advected quantity at cell centers
     **/
    typename cell_array::view_type get( Cajita::Cell, Field::Quantity, Version::Current ) const {
        return _quantity_curr->view();
    };

    /**
     * Return Quantity Field
     * @param Location::Cell
     * @param Field::Quantity
     * @param Version::Next
     * @return Returns view of next advected quantity at cell centers
     **/
    typename cell_array::view_type get( Cajita::Cell, Field::Quantity, Version::Next ) const {
        return _quantity_next->view();
    };

    /**
     * Return I Face Velocity 
     * @param Face<Dim::I>
     * @param Field::Velocity
     * @param Version::Current
     * @return Returns view of current norm velocity magnitude on i faces
     **/
    typename cell_array::view_type get( Cajita::Face<Cajita::Dim::I>, Field::Velocity, Version::Current ) const {
        return _ui_curr->view();
    };

    /**
     * Return I Face Velocity 
     * @param Face<Dim::I>
     * @param Field::Velocity
     * @param Version::Next
     * @return Returns view of next norm velocity magnitude on i faces
     **/
    typename cell_array::view_type get( Cajita::Face<Cajita::Dim::I>, Field::Velocity, Version::Next ) const {
        return _ui_next->view();
    };

    /**
     * Return J Face Velocity 
     * @param Face<Dim::J>
     * @param Field::Velocity
     * @param Version::Current
     * @return Returns view of current norm velocity magnitude on j faces
     **/
    typename cell_array::view_type get( Cajita::Face<Cajita::Dim::J>, Field::Velocity, Version::Current ) const {
        return _uj_curr->view();
    };

    /**
     * Return J Face Velocity 
     * @param Face<Dim::J>
     * @param Field::Velocity
     * @param Version::Next
     * @return Returns view of next norm velocity magnitude on j faces
     **/
    typename cell_array::view_type get( Cajita::Face<Cajita::Dim::J>, Field::Velocity, Version::Next ) const {
        return _uj_next->view();
    };

    /**
     * Make the next version of a field the current one 
     * @param Cajita::Cell
     * @param Field::Quantity
     **/
    void advance( Cajita::Cell, Field::Quantity) {
	_quantity_curr.swap(_quantity_next);
    }

    /**
     * Make the next version of a field the current one 
     * @param Cajita::Face<Cajita::DimI>
     * @param Field::Velocity
     **/
    void advance( Cajita::Face<Cajita::Dim::I>, Field::Velocity) {
	_ui_curr.swap(_ui_next);
    }

    /**
     * Make the next version of a field the current one 
     * @param Cajita::Face<Cajita::Dim::J>
     * @param Field::Velocity
     **/
    void advance( Cajita::Face<Cajita::Dim::J>, Field::Velocity) {
	_uj_curr.swap(_uj_next);
    }

    /**
     * Standard three-deep halo patterns for advected mesh fields 
     */
    std::shared_ptr<halo_type> advection_halo( Cell ) const { return _cell_advection_halo; }
    std::shared_ptr<halo_type> advection_halo( FaceI ) const { return _iface_advection_halo; }
    std::shared_ptr<halo_type> advection_halo( FaceJ ) const { return _jface_advection_halo; }

    /**
     * Gather State Data from Neighbors
     * @param Locationl
     * @param Field
     **/
    void gather( Cajita::Cell, Field::Quantity ) const {
        _cell_advection_halo->gather( ExecutionSpace(), *_quantity_curr );
    };
    void gather( Cajita::Face<Cajita::Dim::I>, Field::Velocity ) const {
        _iface_advection_halo->gather( ExecutionSpace(), *_ui_curr);
    };
    void gather( Cajita::Face<Cajita::Dim::J>, Field::Velocity ) const {
        _jface_advection_halo->gather( ExecutionSpace(), *_uj_curr);
    };


  private:
    // The mesh on which our data items are stored
    std::shared_ptr<mesh_type> _mesh; 

    // Basic long-term quantities stored and advected in the mesh
    std::shared_ptr<cell_array> _quantity_curr, _quantity_next;
    std::shared_ptr<iface_array> _ui_curr, _ui_next;
    std::shared_ptr<jface_array> _uj_curr, _uj_next;

    // Halo communicaiton patterns for the advected quantities
    std::shared_ptr<halo_type> _iface_advection_halo;
    std::shared_ptr<halo_type> _jface_advection_halo;
    std::shared_ptr<halo_type> _cell_advection_halo;

    //std::shared_ptr<halo_type> _cell_pressure_halo;
};

} // namespace CajitaFluids

#endif // CAJITAFLUIDS_PROBLEMMANAGER_HPP
