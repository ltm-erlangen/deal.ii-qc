#ifndef __dealii_qc_configure_qc_h
#define __dealii_qc_configure_qc_h

#include <iostream>
#include <string>

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
    ConfigureQC ( /*const Parameters<dim> &parameters*/ );
    ConfigureQC ( const std::string &filename );
    //~ConfigureQC();

    /**
     * Get mesh file name by assigning mesh_file name to the argument
     */
    void get_mesh( std::string &);

    /**
     * Get mesh file
     */
    std::string get_mesh();

  private:
    void configure_qc( const std::string &filename );
    ParameterHandler prm;

    /**
     * Name of the mesh file for initial qc setup
     */
    std::string mesh_file;

    /**
     * Whether or not to perform initial refinement
     */
    bool do_refinement;

    /**
     * Number of cycles of initial global refinement
     */
    unsigned int n_cycles;

  };

  template< int dim>
  ConfigureQC<dim>::ConfigureQC(  )
  :
  prm(), mesh_file(""), do_refinement(false), n_cycles(0)
  {}

}

#include "configure_qc.impl.h"

#endif // __dealii_qc_configure_qc_h
