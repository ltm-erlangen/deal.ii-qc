
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
     * This class is a minor extension to the parallel::shared::Triangulation of
     * deal.II (that provides a parallel triangulation for which every processor
     * knows about every cell of the global mesh) to allow a thickened ghost
     * cell layer around locally owned cells.
     *
     * @note After creating the triangulation or after coarsening or refinement
     * additionally Triangulation::setup_ghost_cells() must be called to
     * re-adjust the ghost cell layer thickness to the desired.
     */
    template <int dim, int spacedim=dim>
    class Triangulation : public dealii::parallel::shared::Triangulation<dim, spacedim>
    {
    public:

      typedef typename dealii::parallel::shared::Triangulation<dim, spacedim>::Settings Settings;

      /**
       * Constructor.
       *
       *
       * If @p ghost_cell_layer_thickness is negative, there are no artificial
       * cells. If it is non-negative, then for each MPI process there is a
       * unique classification of cells into locally owned, ghost and
       * artificial cells and the extent (or thickness) of the ghost cell layer
       * around locally owned cells can be altered using
       * @p ghost_cell_layer_thickness.
       */
      Triangulation (MPI_Comm mpi_communicator,
                     const typename dealii::Triangulation<dim,spacedim>::MeshSmoothing =
                       (dealii::Triangulation<dim,spacedim>::none),
                     const double ghost_cell_layer_thickness = -1.,
                     const Settings settings = Settings::partition_metis);

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
