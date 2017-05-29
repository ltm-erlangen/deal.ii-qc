
#ifndef __dealii_qc_configure_qc_h
#define __dealii_qc_configure_qc_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>

#include <fstream>
#include <sstream>
#include <utility>
#include <memory>

#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_lumped_vertex.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_vertex.h>
#include <deal.II-qc/configure/geometry/geometry_box.h>
#include <deal.II-qc/configure/geometry/geometry_gmsh.h>
#include <deal.II-qc/potentials/pair_coul_wolf.h>
#include <deal.II-qc/potentials/pair_lj_cut.h>

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
     * Get pair potential type.
     */
    std::string get_pair_potential_type() const;

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
     * Return #ghost_cell_layer_thickness.
     */
    double get_ghost_cell_layer_thickness() const;

    // TODO: take maximum_cutoff_radius from pair potential cutoff radii?
    // maximum_cutoff_radius= max{ cutoff_radii } + skin?
    /**
     * Return #maximum_cutoff_radius.
     */
    double get_maximum_cutoff_radius() const;

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
     * Pair potential type to be used.
     */
    std::string pair_potential_type;

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

  protected:

    /**
     * A shared pointer to the pair potential object.
     */
    mutable std::shared_ptr<Potential::PairBaseManager> pair_potential;

    /**
     * In distributed memory calculation with local h-adaptive FE each
     * MPI process stores a non-overlapping subset of all cells, such that
     * their union gives the global mesh. Additionally each core stores a
     * layer of so called ghost cells. For non-local methods such as QC,
     * the calculation of energy functional and its derivative may require
     * more than a halo layer around locally owned cells. This is due to the
     * non-local energy evaluation for pair potentials where an energy atom
     * within a locally owned cell may interact with another ghost atom
     * located in a cell not owned by this MPI process.
     *
     * The #ghost_cell_layer_thickness is used to build a layer of ghost cells
     * around locally owned cells such that in the reference configuration all
     * atoms that are no further than #ghost_cell_layer_thickness away from the
     * cluster atoms for a given MPI core are located only within the union of
     * locally owned cells and ghost cells. This algorithm relies on the upper
     * bound estimation of the distance between two arbitrary cells using
     * TriaAccessor::enclosing_ball().
     *
     * @note Note that for physically correct results
     * #ghost_cell_layer_thickness should always be more or equal to the
     * maximum cut-off radius used in a pair potential. Since the ghost cells
     * are determined once at the beginning of calculations, the value of this
     * parameter should be large enough to account for expected deformations in
     * the system so that at each step of applied load the ghost cells contain
     * all ghost atoms.
     */
    double ghost_cell_layer_thickness;

    /**
     * The maximum of all the cutoff radii of the pair potentials.
     * It is used to update neighbor lists of atoms.
     *
     * @note #maximum_cutoff_radius is different from
     * #ghost_cell_layer_thickness which is used to identify locally relevant
     * ghost cells of MPI processes.
     */
    double maximum_cutoff_radius;

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
