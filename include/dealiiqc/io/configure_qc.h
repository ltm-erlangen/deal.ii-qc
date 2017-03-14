
#ifndef __dealii_qc_configure_qc_h
#define __dealii_qc_configure_qc_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>

#include <fstream>
#include <sstream>
#include <utility>

#include <dealiiqc/io/parse_atom_data.h>
#include <dealiiqc/utility.h>

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
     * Constructor with istream object
     */
    ConfigureQC( std::istream &);

    /**
     * Constructor with parameter filename as the argument
     */
    ConfigureQC ( const std::string &);

    /**
     * Get dimensionality of the problem
     */
    unsigned int get_dimension() const;

    /**
     * Get current mesh file
     */
    std::string get_mesh_file() const;

    /**
     * Get atom data file
     */
    std::string get_atom_data_file() const;

    /**
     * Get number of initial grid refinement cycles
     */
    unsigned int get_n_initial_global_refinements() const;


    /**
     * Get parse post parameter section bool
     */
    std::string get_input_post_eop_section();

  private:

    /*
     * Declare parameters to configure qc
     */
    static void declare_parameters( ParameterHandler &prm );

    /*
     * Parse parameters
     */
    void parse_parameters( ParameterHandler &prm );

    /**
     * Dimensionality of the problem
     */
    unsigned int dimension;

    /**
     * Name of the mesh file for initial qc setup
     */
    std::string mesh_file;

    /**
     * Number of cycles of initial global refinement
     */
    unsigned int n_initial_global_refinements;

    /**
     * Name of the atom data file
     */
    std::string atom_data_file;

    /**
     * Parse QC configuration file post
     */
    // Not using std::istringstream because in
    // the constructor std::move hits a compiler bug (<gcc 5.0)
    // tried with gcc 5.4.0 error persists

    std::string input_post_eop_section;

  };

}

#endif // __dealii_qc_configure_qc_h
