
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
#include <dealiiqc/potentials/pair_lj_cut.h>
#include <dealiiqc/potentials/pair_coul_wolf.h>
#include <dealiiqc/utilities.h>

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

    /**
     * Get maximum search radius
     */
    double get_maximum_search_radius() const;

    // TODO: take maximum_energy_radius from pair potential cutoff radii?
    // maximum_energy_radius= max{ cutoff_radii } + skin?
    /**
     * Get maximum energy radius
     */
    double get_maximum_energy_radius() const;

    /**
     * Get cluster radius
     */
    double get_cluster_radius() const;

    /**
     * Get a shared pointer to the pair potential class object.
     */
    std::shared_ptr<Potential::PairBaseManager> get_potential() const;

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
     * A shared pointer to the pair potential object.
     */
    mutable std::shared_ptr<Potential::PairBaseManager> pair_potential;

    /**
     * Maximum search distance from any of the vertices of locally owned cells
     * to an atom, to identify whether the atom contributes to the
     * QC energy computations.
     *
     * @p maximum_search_radius is also used to identify ghost cells of a
     * current MPI process. If any of a cell's vertices are within a
     * @p maximum_search_radius distance from any of locally owned cell's vertices,
     * then the cell is a ghost cell of a current MPI process.
     *
     * @note @p maximum_search_radius should not be less than the sum of cluster
     * radius and (maximum) cutoff radius.
     */
    double maximum_search_radius;

    /**
     * @p maximum_energy_radius is the maximum of all the cutoff radii
     * of the pair potentials. It is used to update neighbor lists of
     * atoms.
     *
     * @note @p maximum_search_radius is different from
     * @p maximum_energy_radius. The former is used to identify ghost cells
     * of a current MPI process, while the latter is used to update
     * neighbor lists of atoms.
     */
    double maximum_energy_radius;

    /**
     * @p cluster_radius is the cluster radius used in QC
     */
    double cluster_radius;

  };

}

#endif // __dealii_qc_configure_qc_h
