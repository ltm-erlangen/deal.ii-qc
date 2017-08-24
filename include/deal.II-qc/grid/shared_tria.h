
#ifndef __dealii_qc_shared_tria_h
#define __dealii_qc_shared_tria_h

#include <deal.II-qc/utilities.h>

#include <deal.II/distributed/shared_tria.h>


DEAL_II_QC_NAMESPACE_OPEN


#ifdef DEAL_II_WITH_MPI
namespace parallel
{
  namespace shared
  {

    /**
     * This class is minor extension to the parallel::shared::Triangulation of
     * deal.II (that provides a parallel triangulation for which every processor
     * knows about every cell of the global mesh) to allow a thickened ghost
     * cell layer around locally owned cells.
     *
     * @note After creating the triangulation or after coarsening or refinement
     * additionally Triangulation::setup_ghost_cells() must be called to
     * re-adjust the ghost cell layer thickness to the desired.
     */
    template <int dim, int spacedim>
    class Triangulation : dealii::parallel::shared::Triangulation<dim, spacedim>
    {
    public:

      /**
       * Constructor.
       *
       * If @p allow_aritifical_cells is true, for each MPI process there is a
       * unique classification of cells into locally owned, ghost and artificial
       * cells. In such a case, by default the triangulation on each MPI process
       * has a single layer of ghost cells. However, this behavior can be
       * modified using a positive @p ghost_cell_layer_thickness.
       */
      Triangulation (MPI_Comm mpi_communicator,
                     const typename dealii::Triangulation<dim,spacedim>::MeshSmoothing =
                       (dealii::Triangulation<dim,spacedim>::none),
                     const bool allow_artificial_cells = true,
                     const typename dealii::parallel::shared::Triangulation<dim, spacedim>::Settings settings =
                       dealii::parallel::shared::Triangulation<dim, spacedim>::partition_metis,
                     const double ghost_cell_layer_thickness = 0.);

      // TODO: Need to rework for large deformations?
      /**
       * Re-adjust the ghost cell layer thickness.
       */
      void setup_ghost_cells ();

    private:
      /**
       * Ghost cell layer thickness.
       */
      const double ghost_cell_layer_thickness;

    };
  }
}

#endif
DEAL_II_QC_NAMESPACE_CLOSE




#endif /* __dealii_qc_shared_tria_h */
