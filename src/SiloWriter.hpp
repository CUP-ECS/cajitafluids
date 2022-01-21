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

#ifndef DEBUG
#define DEBUG 1
#endif

// Include Statements
#include <Cajita.hpp>

#ifdef HAVE_SILO
#include <pmpio.h>
#include <silo.h>
#endif

namespace CajitaFluids {

/**
 * The SiloWriter Class
 * @class SiloWriter
 * @brief SiloWriter class to write results to Silo file using PMPIO
 **/
    template <std::size_t Dims, class MemorySpace, class ExecutionSpace>
    class SiloWriter {
      public:
        /**
         * Constructor
         * Create new SiloWriter
         * 
         * @param pm Problem manager object
         */
        template <class ProblemManagerType>
        SiloWriter( ProblemManagerType &pm )
            : _pm( pm ) {
            if ( DEBUG && _pm->mesh()->rank() == 0 ) std::cout << "Created CajitaFluids SiloWriter\n";
        };

        /**
         * Write File
         * @param dbile File handler to dbfile
         * @param name File name
         * @param time_step Current time step
         * @param time Current tim
         * @param dt Time Step (dt)
         **/
        void writeFile( DBfile *dbfile, char *name, int time_step, double time, double dt ) {
            // Initialize Variables
            int        dims[Dims], nx, ny, ndims;
            double    *coords[Dims], *vars[Dims], spacing[Dims];
            char *     coordnames[Dims], *varnames[Dims];
            DBoptlist *optlist;

            // Define device_type for Later Use
            using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

            // Rertrieve the Local Grid and Local Mesh
            auto local_grid = _pm->mesh()->localGrid();
            auto local_mesh = _pm->mesh()->localMesh();

            // DEBUG: Trace Writing File
            if ( DEBUG ) std::cout << "Writing File\n";

            // Set DB Options: Time Step, Time Stamp and Delta Time
            optlist = DBMakeOptlist( 10 );
            DBAddOption( optlist, DBOPT_CYCLE, &time_step );
            DBAddOption( optlist, DBOPT_TIME, &time );
            DBAddOption( optlist, DBOPT_DTIME, &dt );

            // Get Domain Space
            auto domain = _pm->mesh()->domainSpace();

            for (int i = 0; i < Dims; i++) {
	        zdims[i] = domain.extent(i); // zones (cells) in a dimension
                dims[i] = zdims[i] + 1; // nodes in a dimension
                spacing[i] = _pm->mesh()->cellSize();
            }

            // Coordinate Names: Cartesian X, Y Coordinate System
            coordnames[0] = strdup( "x" );
            coordnames[1] = strdup( "y" );
            if (Dims == 3) coordnames[2] = strdup( "z" );

            // Initialize Coordinate and State Arrays for Writing
	    for (int i = 0; i < Dims; i++) {
                coords[i] = (double *)malloc(sizeof(double) * dims[i]);
            }

            // Point Coords to X and Y Coordinates
            if (Dims == 3) coords[2] = z;
            // Set X and Y Coordinates of Nodes
            for ( int d = 0; d < Dims; d++) {
                for ( int i = domain.min( d ); i <= domain.max( d ); i++ ) {
                    int     iown      = i - domain.min( d );
                    int     index[3] = { 0, 0, 0 };
                    state_t location[3];
		    index[d] = i;
                    local_mesh.coordinates( Cajita::Node(), index, location );
                    axis[d][iown] = location[d];
                }
            }

            DBPutQuadmesh( dbfile, name, (DBCAS_t)coordnames,
                           coords, dims, Dims, DB_DOUBLE, DB_COLLINEAR, optlist );

            // Get State Views
            auto q = _pm->get( Cajita::Cell(), Field::Quantity() );
            auto u = _pm->get( Cajita::Face<Cajita::Dim::I>(), Field::Velocity() );
            auto v = _pm->get( Cajita::Face<Cajita::Dim::J>(), Field::Velocity() );
            auto qHost = Kokkos::create_mirror_view( q );
            auto uHost = Kokkos::create_mirror_view( u );
            auto vHost = Kokkos::create_mirror_view( v );
            Kokkos::deep_copy( qHost, q );
            Kokkos::deep_copy( uHost, u );
            Kokkos::deep_copy( vHost, v );
	    // declare w and get it if 3 dimensions
            if constexpr (std::is_same_v<Dim, 3>) {
                auto wHost = Kokkos::create_mirror_view( w );
            } 

            // Write Scalar Variables
            // Quantty
            DBPutQuadvar1( dbfile, "height", name, qHost.data(), zdims, Dims,
                           NULL, 0, DB_DOUBLE, DB_ZONECENT, optlist );

	    
            // Velocity
	    vars[0] = uHost.data();
            vars[1] = vHost.data();
            varnames[0] = strdup( "u" );
            varnames[1] = strdup( "v" );
            if constexpr (std::is_same_v<Dim, 3>) {
                vars[2] = wHost.data();
                varnames[2] = strdup( "w" );
	    } 
            DBPutQuadvar1( dbfile, "velocity", name, vars, zdims, Dims,
                           NULL, 0, DB_DOUBLE, DB_FACECENT, optlist );

            // Initialize Coordinate and State Arrays for Writing
	    for (int i = 0; i < Dims; i++) {
                free(coords[i]);
            }

            // Free Option List
            DBFreeOptlist( optlist );
        };

        /**
         * Create New Silo File for Current Time Step and Owning Group
         * @param filename Name of file
         * @param nsname Name of directory inside of the file
         * @param user_data File Driver/Type (PDB, HDF5)
         **/
        static void *createSiloFile( const char *filename, const char *nsname, void *user_data ) {
            if ( DEBUG ) std::cout << "Creating file: " << filename << "\n";

            int     driver    = *( (int *)user_data );
            DBfile *silo_file = DBCreate( filename, DB_CLOBBER, DB_LOCAL, "ExaCLAMRRaw", driver );

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
        static void *openSiloFile( const char *filename, const char *nsname, PMPIO_iomode_t ioMode, void *user_data ) {
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
            DBfile *silo_file = (DBfile *)file;
            if ( silo_file ) DBClose( silo_file );
        };

        /**
         * Write Multi Object Silo File the References Child Files in order to have entire set of data for the time step within a Single File
         * Combines several Silo Files into a Single Silo File
         * 
         * @param silo_file Pointer to the Silo File
         * @param baton Baton object from PMPIO
         * @param size Number of Ranks
         * @param time_step Current time step
         * @param file_ext File extension (PDB, HDF5)
         **/
        void writeMultiObjects( DBfile *silo_file, PMPIO_baton_t *baton, int size, int time_step, const char *file_ext ) {
            char **mesh_block_names = (char **)malloc( size * sizeof( char * ) );
            char **h_block_names    = (char **)malloc( size * sizeof( char * ) );
            char **u_block_names    = (char **)malloc( size * sizeof( char * ) );
            char **v_block_names    = (char **)malloc( size * sizeof( char * ) );
            char **mom_block_names  = (char **)malloc( size * sizeof( char * ) );

            int *block_types = (int *)malloc( size * sizeof( int ) );
            int *var_types   = (int *)malloc( size * sizeof( int ) );

            DBSetDir( silo_file, "/" );

            for ( int i = 0; i < size; i++ ) {
                int group_rank      = PMPIO_GroupRank( baton, i );
                mesh_block_names[i] = (char *)malloc( 1024 );
                h_block_names[i]    = (char *)malloc( 1024 );
                u_block_names[i]    = (char *)malloc( 1024 );
                v_block_names[i]    = (char *)malloc( 1024 );
                mom_block_names[i]  = (char *)malloc( 1024 );

                sprintf( mesh_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/Mesh", group_rank, time_step, i );
                sprintf( h_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/height", group_rank, time_step, i );
                sprintf( u_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/ucomp", group_rank, time_step, i );
                sprintf( v_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/vcomp", group_rank, time_step, i );
                sprintf( mom_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/momentum", group_rank, time_step, i );

                block_types[i] = DB_QUADMESH;
                var_types[i]   = DB_QUADVAR;
            }

            DBPutMultimesh( silo_file, "multi_mesh", size, mesh_block_names, block_types, 0 );
            DBPutMultivar( silo_file, "multi_height", size, h_block_names, var_types, 0 );
            DBPutMultivar( silo_file, "multi_ucomp", size, u_block_names, var_types, 0 );
            DBPutMultivar( silo_file, "multi_vcomp", size, v_block_names, var_types, 0 );
            DBPutMultivar( silo_file, "multi_momentum", size, mom_block_names, var_types, 0 );

            for ( int i = 0; i < size; i++ ) {
                free( mesh_block_names[i] );
                free( h_block_names[i] );
                free( u_block_names[i] );
                free( v_block_names[i] );
                free( mom_block_names[i] );
            }

            free( mesh_block_names );
            free( h_block_names );
            free( u_block_names );
            free( v_block_names );
            free( mom_block_names );
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
        void siloWrite( char *name, int time_step, state_t time, state_t dt ) {
            // Initalize Variables
            DBfile *silo_file;
            DBfile *master_file;
            int     size;
            int     driver = DB_PDB;
            // TODO: Make the Number of Groups a Constant or a Runtime Parameter ( Between 8 and 64 )
            int            numGroups = 2;
            char           masterfilename[256], filename[256], nsname[256];
            PMPIO_baton_t *baton;

            MPI_Comm_size( MPI_COMM_WORLD, &size );
            MPI_Bcast( &numGroups, 1, MPI_INT, 0, MPI_COMM_WORLD );
            MPI_Bcast( &driver, 1, MPI_INT, 0, MPI_COMM_WORLD );

            baton = PMPIO_Init( numGroups, PMPIO_WRITE, MPI_COMM_WORLD, 1, createSiloFile, openSiloFile, closeSiloFile, &driver );

            // Set Filename to Reflect TimeStep
            sprintf( masterfilename, "data/ExaCLAMR%05d.pdb", time_step );
            sprintf( filename, "data/raw/ExaCLAMROutput%05d%05d.pdb", PMPIO_GroupRank( baton, _pm->mesh()->rank() ), time_step );
            sprintf( nsname, "domain_%05d", _pm->mesh()->rank() );

            // Show Errors and Force FLoating Point
            DBShowErrors( DB_ALL, NULL );

            silo_file = (DBfile *)PMPIO_WaitForBaton( baton, filename, nsname );

            writeFile( silo_file, name, time_step, time, dt );

            if ( _pm->mesh()->rank() == 0 ) {
                master_file = DBCreate( masterfilename, DB_CLOBBER, DB_LOCAL, "ExaCLAMR", driver );
                writeMultiObjects( master_file, baton, size, time_step, "pdb" );
                DBClose( master_file );
            }

            PMPIO_HandOffBaton( baton, silo_file );

            PMPIO_Finish( baton );
        }

      private:
        std::shared_ptr<ProblemManager<ExaCLAMR::RegularMesh<state_t>, MemorySpace, ExecutionSpace, OrderingView>> _pm; /**< Problem Manager Shared Pointer */
    };

}; // namespace ExaCLAMR

#endif
