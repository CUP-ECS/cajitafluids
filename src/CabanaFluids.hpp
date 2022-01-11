
/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * ExaCLAMR Header File: Contains macros used for switching between new and current field views
 * 
 */

/**
 * @namespace ExaCLAMR
 * @brief Contains classes, data structures, and routines to solve shallow water equations on a regular mesh
 **/
namespace ExaCLAMR {
// Squared Macro
#define POW2( x ) ( ( x ) * ( x ) )

// Toggle Between Current and New State Vectors
#define NEWFIELD( time_step )     ( ( time_step + 1 ) % 2 )
#define CURRENTFIELD( time_step ) ( ( time_step ) % 2 )

} // namespace ExaCLAMR
