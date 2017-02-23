
#ifndef __dealii_qc_configure_qc_h
#define __dealii_qc_configure_qc_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>

namespace dealiiqc
{
  using namespace dealii;

  /**
   * A class to read qc input file.
   * The input file should contain the following information:
   * - Initial mesh information
   * - Atoms and atoms attributes
   * - Type of operations
   */
  template <int dim>
  class ConfigureQC
  {
  public:

    /**
     * Default constructor
     */
    ConfigureQC ( /*const Parameters<dim> &parameters*/ );

    /**
     * Constructor with input file name as the argument
     */
    ConfigureQC ( const std::istringstream &iss );

    /**
     * Get current mesh file
     */
    std::string get_mesh_file();

    /**
     * Get number of initial grid refinement cycles
     */
    unsigned int get_n_initial_global_refinements();

  private:
    void configure_qc( const std::istringstream &iss );

    /**
     * ParameterHandler to parse inputfile
     */
    ParameterHandler prm;

    /**
     * Name of the mesh file for initial qc setup
     */
    std::string mesh_file;

    /**
     * Number of cycles of initial global refinement
     */
    unsigned int n_initial_global_refinements;

    // TODO: parse atom data

  };

}

#include "../../../source/io/configure_qc.cc"

#endif // __dealii_qc_configure_qc_h
