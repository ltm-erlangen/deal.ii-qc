
#ifndef __dealii_qc_configure_qc_h
#define __dealii_qc_configure_qc_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>

#include <fstream>
#include <sstream>
#include <utility>
#include <memory>

#include <dealiiqc/atom/cluster_weights.h>
#include <dealiiqc/configure/geometry/geometry_box.h>
#include <dealiiqc/configure/geometry/geometry_gmsh.h>
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
     * Constructor with a shared pointer to an istream object @p is.
     */
    ConfigureQC( std::shared_ptr<std::istream> is);

    /**
     * Get dimensionality of the problem
     */
    unsigned int get_dimension() const;

    /**
     * Get a shared pointer to const dim dimensional Geometry object.
     *
     * The function returns one of #geometry_3d, #geometry_2d or #geometry_1d
     * depending on the dimension provided in the input script while
     * constructing the ConfigureQC object.
     */
    template<int dim>
    std::shared_ptr<const Geometry::Base<dim>> get_geometry () const;

    /**
     * Get atom data file.
     */
    std::string get_atom_data_file() const;

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
     * Get maximum energy radius.
     */
    double get_maximum_energy_radius() const;

    /**
     * Get cluster radius.
     */
    double get_cluster_radius() const;

    /**
     * Get a shared pointer to the pair potential class object.
     */
    std::shared_ptr<Potential::PairBaseManager> get_potential() const;

    /**
     * Create and return a shared pointer to the derived class object of
     * WeightsByBase.
     */
    template <int dim>
    std::shared_ptr<const Cluster::WeightsByBase<dim>> get_cluster_weights() const;

  private:

    /*
     * Declare parameters to configure QC class.
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
     * Shared pointer to the three dimensional Geometry object.
     *
     * @note The dimension of the Geometry object has to be read from the input
     * stream, therefore in ConfigureQC class, shared pointers to all the three
     * dimensioned Geometry object is declared. However, upon reading the input
     * stream only the relevant shared pointer is initialized and the rest of
     * the shared pointers are NULL.
     */
    std::shared_ptr<const Geometry::Base<3>> geometry_3d;

    /**
     * Shared pointer to the two dimensional Geometry object.
     *
     * See note in #geometry_3d
     */
    std::shared_ptr<const Geometry::Base<2>> geometry_2d;

    /**
     * Shared pointer to the one dimensional Geometry object.
     *
     * See note in #geometry_3d
     */
    std::shared_ptr<const Geometry::Base<1>> geometry_1d;

    /**
     * Path to the atom data file.
     */
    std::string atom_data_file;

    /**
     * Shared pointer to the input stream passed in to the
     * constructor ConfigureQC().
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
     * #maximum_search_radius is also used to identify ghost cells of a
     * current MPI process. If any of a cell's vertices are within a
     * #maximum_search_radius distance from any of locally owned cell's vertices,
     * then the cell is a ghost cell of a current MPI process.
     *
     * @note #maximum_search_radius should not be less than the sum of cluster
     * radius and (maximum) cutoff radius.
     */
    double maximum_search_radius;

    /**
     * #maximum_energy_radius is the maximum of all the cutoff radii
     * of the pair potentials. It is used to update neighbor lists of
     * atoms.
     *
     * @note #maximum_search_radius is different from
     * #maximum_energy_radius. The former is used to identify ghost cells
     * of a current MPI process, while the latter is used to update
     * neighbor lists of atoms.
     */
    double maximum_energy_radius;

    /**
     * The cluster radius used in QC.
     */
    double cluster_radius;

    /**
     * The type of method to update cluster weights.
     */
    std::string cluster_weights_type;

  };

}

#endif // __dealii_qc_configure_qc_h
