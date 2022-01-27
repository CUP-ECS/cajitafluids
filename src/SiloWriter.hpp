/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Silo Writer class to write results to a silo file using PMPIO
 */

#ifndef CAJITAFLUIDS_SILOWRITER_HPP
#define CAJITAFLUIDS_SILOWRITER_HPP

//#ifndef DEBUG
#define DEBUG 1
//#endif

// Include Statements
#include <Cajita.hpp>

#include <pmpio.h>
#include <silo.h>

namespace CajitaFluids {

/**
 * The SiloWriter Class
 * @class SiloWriter
 * @brief SiloWriter class to write results to Silo file using PMPIO
 **/
template <std::size_t Dims, class ExecutionSpace, class MemorySpace>
class SiloWriter {
  public:
    using pm_type = ProblemManager<Dims, ExecutionSpace, MemorySpace>;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    /**
     * Constructor
     * Create new SiloWriter
     * 
     * @param pm Problem manager object
     */
    template <class ProblemManagerType>
    SiloWriter( ProblemManagerType &pm )
        : _pm( pm ) {
        if ( DEBUG && _pm->mesh()->rank() == 0 ) std::cerr << "Created CajitaFluids SiloWriter\n";
    };

    /**
     * Write File
     * @param dbile File handler to dbfile
     * @param name File name
     * @param time_step Current time step
     * @param time Current tim
     * @param dt Time Step (dt)
     * @brief Writes the locally-owned portion of the mesh/variables to a file
     **/
    void writeFile( DBfile *dbfile, char *meshname, int time_step, double time, double dt ) {
        // Initialize Variables
        int        dims[Dims], zdims[Dims];
        double    *coords[Dims], *vars[Dims], spacing[Dims];
	const char *coordnames[3] = { "X", "Y", "Z"};
        DBoptlist *optlist;

        // Rertrieve the Local Grid and Local Mesh
        auto local_grid = _pm->mesh()->localGrid();
        auto local_mesh = *(_pm->mesh()->localMesh());

        if ( DEBUG ) std::cerr << "Writing File\n";

        // Set DB Options: Time Step, Time Stamp and Delta Time
        optlist = DBMakeOptlist( 10 );
        DBAddOption( optlist, DBOPT_CYCLE, &time_step );
        DBAddOption( optlist, DBOPT_TIME, &time );
        DBAddOption( optlist, DBOPT_DTIME, &dt );

        // Get the size of the local cell space and declare the 
	// coordinates of the portion of the mesh we're writing
        auto cell_domain = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

        for (int i = 0; i < Dims; i++) {
            zdims[i] = cell_domain.extent(i); // zones (cells) in a dimension
            dims[i] = zdims[i] + 1; // nodes in a dimension
            spacing[i] = _pm->mesh()->cellSize(); // uniform mesh
        }

        // Coordinate Names: Cartesian X/Y/Z Coordinate System
	for (int i = 0; i < Dims; i++) {
	}

        // Allocate coordinate arrays in each dimension
	for (int i = 0; i < Dims; i++) {
            coords[i] = (double *)malloc(sizeof(double) * dims[i]);
        }

        // Fill out coords[] arrays with coordinate values in each dimension
        for ( int d = 0; d < Dims; d++) {
	    if (DEBUG) std::cerr << "Writing coords for dim " << d << " for range "
                                 << cell_domain.min(d) << " to " << cell_domain.max(d) << "\n";
            for ( int i = cell_domain.min( d ); 
		      i < cell_domain.max( d ) + 1; i++ ) {
                int     iown      = i - cell_domain.min( d );
                int     index[Dims]; 
                double location[Dims];
	        for (int j = 0; j < Dims; j++) index[j] = 0;
	        index[d] = i;
                local_mesh.coordinates( Cajita::Node(), index, location );
                coords[d][iown] = location[d];
            }
        }

        if ( DEBUG ) std::cerr << "Writing quadmesh description\n";
        DBPutQuadmesh( dbfile, meshname, (DBCAS_t)coordnames,
                       coords, dims, Dims, DB_DOUBLE, DB_COLLINEAR, optlist );

	// Now we write the individual variables assocuated with this 
	// portion of the mesh, potentially copying them out of device space
 	// and making sure not to write ghost values.

        // Advected quantity first - copy owned portion from the primary 
        // execution space to the host execution space
        auto q = _pm->get( Cajita::Cell(), Field::Quantity() );
	auto xmin = cell_domain.min(0);
	auto ymin = cell_domain.min(1);
        Kokkos::View<typename pm_type::cell_array::value_type***, Kokkos::LayoutLeft,
		     typename pm_type::cell_array::device_type> 
            qOwned("qowned", cell_domain.extent(0), cell_domain.extent(1), 1);
	Kokkos::parallel_for( 
            "SiloWriter::qowned copy", 
            createExecutionPolicy( cell_domain, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
	        qOwned(i - xmin, j - ymin, 0) = q(i, j, 0);
            });
        auto qHost = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), qOwned );

        if ( DEBUG ) std::cerr << "Writing quantity variable\n";
        DBPutQuadvar1( dbfile, "quantity", meshname, qHost.data(), zdims, Dims,
                       NULL, 0, DB_DOUBLE, DB_ZONECENT, optlist );

        // Velocity faces need to be in a single array, ordered by i then j
	// edges. i-faces in 2D are j-oriented edges, so we do v then u. In 
        // addition, each subportion of this array is also padded out to be 
	// dim[0] by dim[1] in size for Silo.
        
	double *velocities = (double *)malloc(sizeof(double) * dims[0] * dims[1] * Dims );

	// Copy the v velocity face values to a row-major view on the 
	// host and then into the array. Note that we have to pad this out
	// to be dims[0] * dims[1] in size.
        if ( DEBUG ) std::cerr << "Working on v velocity variable.\n";
        auto v = _pm->get( Cajita::Face<Cajita::Dim::J>(), Field::Velocity() );
        Kokkos::View<typename pm_type::jface_array::value_type***, Kokkos::LayoutLeft,
		     typename pm_type::jface_array::device_type> 
            vOwned("vowned", dims[0], dims[1], 1);
        auto jface_domain = local_grid->indexSpace( Cajita::Own(), 
            Cajita::Face<Cajita::Dim::J>(), Cajita::Local() );
	Kokkos::parallel_for( 
            "SiloWriter::vowned copy", 
            createExecutionPolicy( jface_domain, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
	        vOwned(i - xmin, j - ymin, 0) = v(i, j, 0);
            });
        auto vHost = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), 
                                                          vOwned );
        if ( DEBUG ) std::cerr << "Copying in v velocity variable\n";
	memcpy(velocities, vHost.data(), dims[0] * dims[1] * sizeof(double));

	// Now move the v velocity faces to the array the same way.
        if ( DEBUG ) std::cerr << "Working on u velocity variable.\n";
        auto u = _pm->get( Cajita::Face<Cajita::Dim::I>(), Field::Velocity() );

	// The view is size dims[0] by dims[1] but the parallel loop to
	// copy into it iterates over the space of edge indicies
        Kokkos::View<typename pm_type::iface_array::value_type***, Kokkos::LayoutLeft,
		     typename pm_type::iface_array::device_type> 
            uOwned("uowned", dims[0], dims[1], 1);
        auto iface_domain = local_grid->indexSpace( Cajita::Own(), 
            Cajita::Face<Cajita::Dim::I>(), Cajita::Local() );
	Kokkos::parallel_for( 
            "SiloWriter::uowned copy", 
            createExecutionPolicy( iface_domain, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
	        uOwned(i - xmin, j - ymin, 0) = u(i, j, 0);
            });
        auto uHost = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), 
							  uOwned);
        if ( DEBUG ) std::cerr << "Copying in u velocity variable\n";
	memcpy(velocities + dims[0] * dims[1], uHost.data(), 
               dims[0] * dims[1] * sizeof(double));

	// Finally write the velocity faces to the output file. Again, this is 
        // true edge-centered data, so it wants the number of nodes, not zones.
        if ( DEBUG ) std::cerr << "Writing velocity variable to silo quadmesh\n";
        DBPutQuadvar1( dbfile, "velocity", meshname, velocities, dims, Dims,
                       NULL, 0, DB_DOUBLE, DB_EDGECENT, optlist );

        // Free Coordinate and State Arrays for Writing
	free(velocities);
	for (int i = 0; i < Dims; i++) {
            free(coords[i]);
        }

        // Free Option List
        DBFreeOptlist( optlist );
        if ( DEBUG ) std::cerr << "Finished writing variables to file\n";
    };

    /**
     * Create New Silo File for Current Time Step and Owning Group
     * @param filename Name of file
     * @param nsname Name of directory inside of the file
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void *createSiloFile( const char *filename, const char *nsname, void *user_data ) {
        if ( DEBUG ) std::cerr << "Creating file: " << filename << "\n";

        int     driver    = *( (int *)user_data );
        DBfile *silo_file = DBCreate( filename, DB_CLOBBER, DB_LOCAL, "CajitaFluidsRaw", driver );

        if ( silo_file ) {
            DBMkDir( silo_file, nsname );
            DBSetDir( silo_file, nsname );
        }

        return (void *)silo_file;
    };

    /**
     * Open Silo File
     * @param filename Name of file
     * @param nsname Name of directory inside of file
     * @param ioMode Read/Write/Append Mode
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void *openSiloFile( const char *filename, const char *nsname, 
			       PMPIO_iomode_t ioMode, void *user_data ) {
        if ( DEBUG ) std::cerr << "SiloWriter: opening Silo file " << filename << "\n"; 
        DBfile *silo_file = DBOpen( filename, DB_UNKNOWN, ioMode == PMPIO_WRITE ? DB_APPEND : DB_READ );

        if ( silo_file ) {
            if ( ioMode == PMPIO_WRITE ) {
                DBMkDir( silo_file, nsname );
            }
            DBSetDir( silo_file, nsname );
        }

        return (void *)silo_file;
    };

    /**
     * Close Silo File
     * @param file File pointer
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void closeSiloFile( void *file, void *user_data ) {
        if ( DEBUG ) std::cerr << "SiloWriter: closing Silo file.\n"; 
        DBfile *silo_file = (DBfile *)file;
        if ( silo_file ) DBClose( silo_file );
    };

    /**
     * Write Multi Object Silo File the References Child Files in order to 
     * have entire set of data for the time step writen by each rank in
     * a single logical file
     * 
     * @param silo_file Pointer to the Silo File
     * @param baton Baton object from PMPIO
     * @param size Number of Ranks
     * @param time_step Current time step
     * @param file_ext File extension (PDB, HDF5)
     **/
    void writeMultiObjects( DBfile *silo_file, PMPIO_baton_t *baton, int size, int time_step, const char *file_ext ) {
        char **mesh_block_names = (char **)malloc( size * sizeof( char * ) );
        char **q_block_names    = (char **)malloc( size * sizeof( char * ) );
        char **v_block_names    = (char **)malloc( size * sizeof( char * ) );

        int *block_types = (int *)malloc( size * sizeof( int ) );
        int *var_types   = (int *)malloc( size * sizeof( int ) );

        DBSetDir( silo_file, "/" );

        for ( int i = 0; i < size; i++ ) {
            int group_rank      = PMPIO_GroupRank( baton, i );
            mesh_block_names[i] = (char *)malloc( 1024 );
            q_block_names[i]    = (char *)malloc( 1024 );
            v_block_names[i]    = (char *)malloc( 1024 );

            sprintf( mesh_block_names[i], "raw/CajitaFluidsOutput%05d%05d.pdb:/domain_%05d/Mesh", group_rank, time_step, i );
            sprintf( q_block_names[i], "raw/CajitaFluidsOutput%05d%05d.pdb:/domain_%05d/quantity", group_rank, time_step, i );
            sprintf( v_block_names[i], "raw/CajitaFluidsOutput%05d%05d.pdb:/domain_%05d/velocity", group_rank, time_step, i );
            block_types[i] = DB_QUADMESH;
            var_types[i]   = DB_QUADVAR;
        }

        DBPutMultimesh( silo_file, "multi_mesh", size, mesh_block_names, block_types, 0 );
        DBPutMultivar( silo_file, "multi_quantity", size, q_block_names, var_types, 0 );
        DBPutMultivar( silo_file, "multi_velocity", size, v_block_names, var_types, 0 );
        for ( int i = 0; i < size; i++ ) {
            free( mesh_block_names[i] );
            free( q_block_names[i] );
            free( v_block_names[i] );
        }

        free( mesh_block_names );
        free( q_block_names );
        free( v_block_names );
        free( block_types );
        free( var_types );
    }

    // Function to Create New DB File for Current Time Step
    /**
     * Createe New DB File for Current Time Step
     * @param name Name of directory in silo file
     * @param time_step Current time step
     * @param time Current time
     * @param dt Time step (dt)
     **/
    void siloWrite( char *name, int time_step, double time, double dt ) {
        // Initalize Variables
        DBfile *silo_file;
        DBfile *master_file;
        int     size;
        int     driver = DB_PDB;
        // TODO: Make the Number of Groups a Constant or a Runtime Parameter ( Between 8 and 64 )
        int            numGroups = 2;
        char           masterfilename[256], filename[256], nsname[256];
        PMPIO_baton_t *baton;

        if ( DEBUG && _pm->mesh()->rank() == 0 ) std::cerr << "SiloWriter: Initializing PMPIO.\n"; 

        MPI_Comm_size( MPI_COMM_WORLD, &size );
        MPI_Bcast( &numGroups, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &driver, 1, MPI_INT, 0, MPI_COMM_WORLD );

        baton = PMPIO_Init( numGroups, PMPIO_WRITE, MPI_COMM_WORLD, 1, createSiloFile, openSiloFile, closeSiloFile, &driver );

        // Set Filename to Reflect TimeStep
        sprintf( masterfilename, "data/CajitaFluids%05d.pdb", time_step );
        sprintf( filename, "data/raw/CajitaFluidsOutput%05d%05d.pdb", PMPIO_GroupRank( baton, _pm->mesh()->rank() ), time_step );
        sprintf( nsname, "domain_%05d", _pm->mesh()->rank() );

        // Show Errors and Force FLoating Point
        DBShowErrors( DB_ALL, NULL );

        if ( DEBUG && _pm->mesh()->rank() == 0 ) std::cerr << "SiloWriter: Waiting for PMPIO file baton.\n"; 
        silo_file = (DBfile *)PMPIO_WaitForBaton( baton, filename, nsname );

        if ( DEBUG && _pm->mesh()->rank() == 0 ) std::cerr << "SiloWriter: Writing to silo file.\n"; 
        writeFile( silo_file, name, time_step, time, dt );

        if ( _pm->mesh()->rank() == 0 ) {
            if ( DEBUG && _pm->mesh()->rank() == 0 ) std::cerr << "SiloWriter: Rank 0 creating master file.\n"; 
            master_file = DBCreate( masterfilename, DB_CLOBBER, DB_LOCAL, "CajitaFluids", driver );
            writeMultiObjects( master_file, baton, size, time_step, "pdb" );
            DBClose( master_file );
        }

        if ( DEBUG && _pm->mesh()->rank() == 0 ) std::cerr << "SiloWriter: Handing off PMPIO baton.\n"; 
        PMPIO_HandOffBaton( baton, silo_file );

        if ( DEBUG && _pm->mesh()->rank() == 0 ) std::cerr << "SiloWriter: Calling PMPIO_Finish.\n"; 
        PMPIO_Finish( baton );
    }

  private:
    std::shared_ptr<pm_type> _pm; 
};

}; // namespace CajitaFluids
#endif
