
#ifndef __dealii_qc_configure_qc_h
#define __dealii_qc_configure_qc_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>

#include <fstream>

namespace dealiiqc
{
  using namespace dealii;

  /**
   * A class to read qc input parameter file.
   * The input parameter file should contain the following information:
   * - Initial mesh information
   * - Atoms attributes
   * - Problem dependent operations
   */
  class ConfigureQC
  {
  public:

    /**
     * Constructor with parameter filename as the argument
     */
     ConfigureQC ( const std::string & );

    /**
     * Get current mesh file
     */
    std::string get_mesh_file() const;

    /**
     * Get number of initial grid refinement cycles
     */
    unsigned int get_n_initial_global_refinements() const;

    /*
     * Declare parameters to configure qc
     */
    static void declare_parameters( ParameterHandler &prm );

    /*
     * Parse parameters
     */
    void parse_parameters( ParameterHandler &prm );

  private:

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

#endif // __dealii_qc_configure_qc_h
