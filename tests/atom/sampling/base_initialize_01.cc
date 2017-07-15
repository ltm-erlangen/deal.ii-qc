
#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>
#include <deal.II-qc/utilities.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>

using namespace dealiiqc;



template<int dim>
class Test : public Cluster::WeightsByBase<dim>
{
public:

  Test (const double cluster_radius,
        const double maximum_cutoff_radius)
    :
    Cluster::WeightsByBase<dim> (cluster_radius,
                                 maximum_cutoff_radius),
    triangulation (MPI_COMM_WORLD,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices),
    mpi_communicator(MPI_COMM_WORLD),
    pcout (std::cout,
           (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0))
  {
    std::vector<unsigned int> repetitions;
    repetitions.push_back(2);
    for (int i = 1; i < dim; ++i)
      repetitions.push_back(1);

    dealii::Point<dim> p1, p2;

    for (int d = 0; d < dim; ++d)
      p2[d] = 1.;

    p2[0] = 2.;

    // +-----+-----+
    // |     |     |
    // |     |     |
    // |     |     |
    // +-----+-----+
    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               repetitions,
                                               p1,
                                               p2,
                                               true);

    triangulation.begin_active()->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();

    // +--+--+-----+
    // |  |  |     |
    // +--+--+     |
    // |  |  |     |
    // +--+--+-----+
  }

  void run ()
  {
    Cluster::WeightsByBase<dim>::initialize (triangulation, QTrapez<dim>());

    pcout << "Total number of vertices: "
          << Cluster::WeightsByBase<dim>::sampling_points.size()
          << std::endl;

    for (const auto &entry : Cluster::WeightsByBase<dim>::cells_to_sampling_indices)
      pcout << entry.first << ":" << entry.second.size() << std::endl;
  }

  dealiiqc::types::CellMoleculeContainerType<dim>
  update_cluster_weights
  (const DoFHandler<dim>                                 &mesh,
   const dealiiqc::types::CellMoleculeContainerType<dim> &cell_molecules) const
  {
    dealiiqc::types::CellMoleculeContainerType<dim> dummy;
    return dummy;
  }

private:
  parallel::shared::Triangulation<dim> triangulation;
  MPI_Comm mpi_communicator;
  ConditionalOStream   pcout;
};



int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize
      mpi_initialization (argc,
                          argv,
                          dealii::numbers::invalid_unsigned_int);

      Test<2> test_2(2., 2.);
      test_2.run();

      Test<3> test_3(2., 2.);
      test_3.run();

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }

  return 0;
}
