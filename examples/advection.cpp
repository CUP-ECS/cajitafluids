/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Simple fluid advection in a domain closed in Y and open in X
 * with a single input source.
 */

#ifndef DEBUG
#define DEBUG 1
#endif

// Include Statements
#include <BoundaryConditions.hpp>
#include <Solver.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#if DEBUG
#include <iostream>
#endif

// Include Statements
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <stdlib.h>


using namespace CajitaFluids;

// Short Args: n - Cell Count, s - Domain Size, 
// x - On-node Parallelism ( Serial/Threaded/OpenMP/CUDA ), 
// t - Time Steps, w - Write Frequency
// d - Input Density, i - Input Location, v - Inflow Velocity
static char *shortargs = ( char * )"n:s:x:t:w:d:i:v:g:";

/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by command line arguments
 */
struct ClArgs {
    int         write_freq; /**< Write frequency */
    std::string device;     /**< ( Serial, Threads, OpenMP, CUDA ) */
    std::array<int, 2>      global_num_cells;  /**< Number of cells */
    Kokkos::Array<double, 4> global_bounding_box; /**< Bounding box of domain */
    std::array<double, 2> inLocation;/**< Inflow Location */
    std::array<double, 2> inSize;    /**< Inflow Size */
    std::array<double, 2> inVelocity;/**< Inflow Velocity */
    double       inQuantity;/**< Inflow Rate */
    double       density;   /**< Density of the fluid */
    double 	 gravity;   /**< Gravity */
    double       delta_t;   /**< Timestep */
    double       t_final;   /**< Ending time */
};

/**
 * Outputs help message explaining command line options.
 * @param rank The rank calling the function
 * @param progname The name of the program
 */
void help( const int rank, char *progname ) {
    if ( rank == 0 ) {
        std::cout << "Usage: " << progname << "\n";
        std::cout << std::left << std::setw( 10 ) << "-x" 
		  << std::setw( 40 ) << "On-node Parallelism Model (default serial)" 
		  << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-s"
		  << std::setw( 40 ) << "Size of domain (default 1.0 1.0)" 
		  << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-n" 
		  << std::setw( 40 ) << "Number of Cells (default 128 128)" 
		  << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-t" 
		  << std::setw( 40 ) << "Amount of time to simulate (default 4.0)" 
		  << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-w" << std::setw( 40 ) 
		  << "Write Frequency (default 20)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-i" << std::setw( 40 ) 
		  << "Inflow Location (default 25, 5)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-v" << std::setw( 40 ) 
		  << "Inflow Velocity (default 0, 10)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-d" << std::setw( 40 ) 
		  << "Inflow Density (default 5)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-d" << std::setw( 40 ) 
		  << "Inflow Density (default 5)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-h" 
		  << std::setw( 40 ) << "Print Help Message" << std::left << "\n";
    }
}

/**
 * Outputs usage hint if invalid command line arguments are given.
 * @param rank The rank calling the function
 * @param progname The name of the program
 */
void usage( const int rank, char *progname ) {
    if ( rank == 0 )  {
        std::cout << "usage: " << progname << " [-s size-of-domain] [-h help]"
                                   << " [-m threading] [-n number-of-cells] [-p periodicity] [-s sigma] [-t number-time-steps] [-w write-frequency]\n";
    }
}

/**
 * Parses command line input and updates the command line variables accordingly.\n
 * Usage: ./[program] [-a halo-size] [-b mesh-type] [-d size-of-domain] [-g gravity] [-h help] [-m threading] [-n number-of-cells] [-o ordering] [-p periodicity] [-s sigma] [-t number-time-steps] [-w write-frequency]
 * @param rank The rank calling the function
 * @param argc Number of command line options passed to program
 * @param argv List of command line options passed to program
 * @param cl Command line arguments structure to store options
 * @return Error status
 */
int parseInput( const int rank, const int argc, char **argv, ClArgs &cl ) {
    cl.device = "serial";              // Default Thread Setting
    cl.t_final = 4.0;   
    cl.delta_t = 0.005;  
    cl.write_freq = 20;  
    cl.global_num_cells    = { 128, 128 };
    cl.global_bounding_box = { 0, 0, 1.0, 1.0 };
    cl.inLocation =  	     {0.45, 0.2};
    cl.inSize =  	     {0.1, 0.01};
    cl.inVelocity =          { 1.0, 0.0 };
    cl.inQuantity = 	     3.0;
    cl.density = 0.1;
    cl.gravity = 0.0;

    // Set Cell Count and Bounding Box Arrays

    // Return Successfully
    return 0;
}

// Initialize field to a constant quantity and velocity
template <std::size_t Dim>
struct MeshInitFunc {
    // Initialize Variables
    double _q, _u[Dim];

    MeshInitFunc( double q, std::array<double, Dim> u)
	: _q(q)
    {
	_u[0] = u[0];
	_u[1] = u[1];
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cajita::Cell, CajitaFluids::Field::Quantity, 
		     const int index[Dim], const double x[Dim], 
	             double &quantity ) const {
	quantity = _q;

        return true;
    };
    KOKKOS_INLINE_FUNCTION
    bool operator()( Cajita::Face<Cajita::Dim::I>, CajitaFluids::Field::Velocity, 
		     const int index[Dim], const double x[Dim], 
	             double &xvelocity ) const {
	xvelocity = _u[0];
        return true;
    };
    KOKKOS_INLINE_FUNCTION
    bool operator()( Cajita::Face<Cajita::Dim::J>, CajitaFluids::Field::Velocity,
		     const int index[Dim], const double x[Dim], 
	             double &yvelocity ) const {
	yvelocity = _u[1];
        return true;
    }
#if 0
    KOKKOS_INLINE_FUNCTION
    bool operator()( Cajita::Face<Cajita::Dim::K>, const int coords[Dim], 
		     const int x[Dim], 
	             double &zvelocity ) const {
	zvelocity = _uz;
        return true;
    }
#endif
};

// Create Solver and Run CLAMR
void advect( ClArgs &cl ) {
    int comm_size, rank;                         // Initialize Variables
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // Get My Rank

    int x_ranks = comm_size;
    while ( x_ranks % 2 == 0 && x_ranks > 2 ) {
        x_ranks /= 2;
    }
    int y_ranks = comm_size / x_ranks;
    if ( DEBUG ) std::cout << "X Ranks: " << x_ranks << " Y Ranks: " << y_ranks << "\n";

    std::array<int, 2> ranks_per_dim = { x_ranks, y_ranks }; // Ranks per Dimension

    Cajita::ManualBlockPartitioner<2> partitioner( ranks_per_dim ); // Create Cajita Partitioner
    CajitaFluids::BoundaryCondition<2> bc;
    bc.boundary_type = {CajitaFluids::BoundaryType::SOLID,
                        CajitaFluids::BoundaryType::SOLID,
		        CajitaFluids::BoundaryType::SOLID,
                        CajitaFluids::BoundaryType::SOLID};

    CajitaFluids::InflowSource<2> source(cl.inLocation, cl.inSize, 
				         cl.inVelocity, cl.inQuantity);
    CajitaFluids::BodyForce<2> body(0.0, -cl.gravity);

    MeshInitFunc<2> initializer( 0.0, {0.0, 0.0});
    auto solver = CajitaFluids::createSolver( cl.device, MPI_COMM_WORLD, 
        cl.global_bounding_box, cl.global_num_cells, partitioner,
        cl.density, initializer, bc, source, body, cl.delta_t) ;
    // Solve
    solver->solve( cl.t_final, cl.write_freq );
}

int main( int argc, char *argv[] ) {
    MPI_Init( &argc, &argv );         // Initialize MPI
    Kokkos::initialize( argc, argv ); // Initialize Kokkos

    // MPI Info
    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // My Rank

    // Parse Input
    ClArgs cl;
    if ( parseInput( rank, argc, argv, cl ) != 0 ) return -1;

    // Only Rank 0 Prints Command Line Options
    if ( rank == 0 ) {
        // Print Command Line Options
        std::cout << "ExaClamr\n";
        std::cout << "=======Command line arguments=======\n";
        std::cout << std::left << std::setw( 20 ) << "Thread Setting" << ": " 
		  << std::setw( 8 ) << cl.device << "\n"; // Threading Setting
        std::cout << std::left << std::setw( 20 ) << "Cells" << ": " 
		  << std::setw( 8 ) << cl.global_num_cells[0] 
		  << std::setw( 8 ) << cl.global_num_cells[1] << "\n"; // Number of Cells
        std::cout << std::left << std::setw( 20 ) << "Domain" << ": "
		  << std::setw( 8 ) << cl.global_bounding_box[2] 
		  << std::setw( 8 ) << cl.global_bounding_box[3] << "\n"; // Span of Domain
        std::cout << std::left << std::setw( 20 ) << "Input Flow" << ": "
		  << std::setw( 8 ) << cl.inQuantity << " at "  
		  << "Location (" << std::setw( 8 ) << cl.inLocation[0] 
		  << std::setw( 8 ) << cl.inLocation[1] << " ) "
		  << "Size (" << std::setw( 8 ) << cl.inSize[0] 
		  << std::setw( 8 ) << cl.inSize[1] << " ) "
		  << "Velocity (" << std::setw( 8 ) << cl.inVelocity[0] 
		  << std::setw( 8 ) << cl.inVelocity[1] 
		  << " )\n"; 
        std::cout << std::left << std::setw( 20 ) << "Total Simulation Time" << ": " 
		  << std::setw( 8 ) << cl.t_final << "\n"; 
        std::cout << std::left << std::setw( 20 ) << "Timestep Size" << ": " 
		  << std::setw( 8 ) << cl.delta_t << "\n"; 
        std::cout << std::left << std::setw( 20 ) << "Write Frequency" << ": " 
		  << std::setw( 8 ) << cl.write_freq << "\n"; // Steps between write
        std::cout << "====================================\n";
    }

    // Call advection solver
    advect( cl );

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI

    return 0;
};
