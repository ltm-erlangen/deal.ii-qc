
#ifndef __dealii_qc_configure_qc_h
#define __dealii_qc_configure_qc_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>

#include <fstream>
#include <sstream>
#include <utility>
#include <memory>

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
    ConfigureQC( std::shared_ptr<std::istream> );

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
     * Get input stream
     */
    std::shared_ptr<std::istream> get_stream() const;

    /*
     * Get pair style
     */
    std::string get_pair_style() const;

    /**
     * Get max search radius
     */
    double get_max_search_radius() const;

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
     * Path to the mesh file for initial qc setup.
     */
    std::string mesh_file;

    /**
     * Number of cycles of initial global refinement
     */
    unsigned int n_initial_global_refinements;

    /**
     * Path to the atom data file.
     */
    std::string atom_data_file;

    /**
     * Shared pointer to the input stream passed in to the
     * constructor @see ConfigureQC().
     */
    mutable std::shared_ptr<std::istream> input_stream;

    /**
     * Pair style
     */
    std::string pair_style;

    /**
     * Maximum search radius to identify ghost layer of subdomains
     */
    double max_search_radius;

  };

}

#endif // __dealii_qc_configure_qc_h
