
#include <deal.II/base/utilities.h>

#include <deal.II-qc/atom/sampling/cluster_weights_by_vertex.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{


  template<int dim, int atomicity, int spacedim>
  WeightsByVertex<dim, atomicity, spacedim>::
  WeightsByVertex (const double &cluster_radius,
                   const double &maximum_cutoff_radius)
    :
    WeightsByBase<dim, atomicity, spacedim> (cluster_radius,
                                             maximum_cutoff_radius)
  {}



  template<int dim, int atomicity, int spacedim>
  types::CellMoleculeContainerType<dim, atomicity, spacedim>
  WeightsByVertex<dim, atomicity, spacedim>::
  update_cluster_weights
  (const dealii::DoFHandler<dim, spacedim>                          &mesh,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules) const
  {
    // Prepare energy molecules in this container.
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    cell_energy_molecules;

    const unsigned int n_vertices = mesh.get_triangulation().n_vertices();

    const parallel::Triangulation<dim, spacedim> *const ptria =
      dynamic_cast<const parallel::Triangulation<dim, spacedim> *>
      (&mesh.get_triangulation());

    const MPI_Comm &mpi_communicator = ptria != nullptr
                                       ?
                                       ptria->get_communicator()
                                       :
                                       MPI_COMM_SELF;

    // Prepare the total number of molecules per vertex in this container.
    // The container should also contain the information of the total number
    // of molecules per per vertex for ghost cells on the current MPI process.
    std::vector<unsigned int> n_molecules_per_vertex(n_vertices,0);

    // Prepare the number of cluster molecules per vertex in this container.
    std::vector<unsigned int> n_cluster_molecules_per_vertex(n_vertices,0);

    // Get the squared_energy_radius to identify energy molecules.
    const double squared_energy_radius =
      dealii::Utilities::fixed_power<2>
      (WeightsByBase<dim, atomicity, spacedim>::maximum_cutoff_radius +
       WeightsByBase<dim, atomicity, spacedim>::cluster_radius);

    // Get the squared_cluster_radius to identify cluster molecules.
    const double squared_cluster_radius =
      dealii::Utilities::fixed_power<2>
      (WeightsByBase<dim, atomicity, spacedim>::cluster_radius);

    // Loop over all molecules, see if a given molecules is energy molecules and
    // if so if it's a cluster molecules.
    // While there, count the total number of molecules per vertex and
    // number of cluster molecules per vertex.
    for (const auto &cell_molecule : cell_molecules)
      {
        const auto &cell = cell_molecule.first;
        Molecule<spacedim, atomicity> molecule = cell_molecule.second;

        // Get the closest vertex (of this cell) to the molecules.
        const auto vertex_and_squared_distance =
          Utilities::find_closest_vertex (molecule_initial_location(molecule),
                                          cell);

        const unsigned int global_vertex_index =
          cell->vertex_index(vertex_and_squared_distance.first);

        const double &squared_distance_from_closest_vertex =
          vertex_and_squared_distance.second;

        // Count only molecules assigned to locally owned cells, so that
        // when sum is performed over all MPI processes only the molecules
        // assigned locally owned cells are counted.
        if (cell->is_locally_owned())
          n_molecules_per_vertex[global_vertex_index]++;

        if (squared_distance_from_closest_vertex < squared_energy_radius)
          {
            if (squared_distance_from_closest_vertex < squared_cluster_radius)
              {
                // Count only the cluster molecules of the locally owned cells.
                if (cell->is_locally_owned())
                  // Increment cluster molecules count for this "vertex"
                  n_cluster_molecules_per_vertex[global_vertex_index]++;
                // molecules is cluster molecules
                molecule.cluster_weight = 1.;
              }
            else
              // molecules is not cluster molecules
              molecule.cluster_weight = 0.;

            // Insert molecules into cell_energy_molecules if it is within a distance of
            // energy_radius to associated cell's vertices.
            cell_energy_molecules.insert(std::make_pair(cell,molecule));
          }
      }

    //---Finished adding energy molecules

    // Accumulate the number of molecules per vertex from all MPI processes.
    dealii::Utilities::MPI::sum (n_molecules_per_vertex,
                                 mpi_communicator,
                                 n_molecules_per_vertex);

    // Accumulate the number of cluster molecules per vertex from all MPI processes.
    dealii::Utilities::MPI::sum (n_cluster_molecules_per_vertex,
                                 mpi_communicator,
                                 n_cluster_molecules_per_vertex);

    //---Now update cluster weights with correct value

    // Loop over all the energy molecules,
    // update their weights by multiplying with the factor
    // (n_molecules/n_cluster_molecules)
    for (auto &energy_molecule : cell_energy_molecules)
      {
        const auto &cell = energy_molecule.first;
        Molecule<spacedim, atomicity>  &molecule = energy_molecule.second;

        // Get the closest vertex (of this cell) to the molecules.
        const auto vertex_and_squared_distance =
          Utilities::find_closest_vertex (molecule_initial_location(molecule),
                                          cell);

        const unsigned int global_vertex_index =
          cell->vertex_index(vertex_and_squared_distance.first);

        Assert (n_cluster_molecules_per_vertex[global_vertex_index] != 0,
                ExcInternalError());

        // The cluster weight was previously set to 1. if the molecules is
        // cluster molecules and 0. if the molecules is not cluster molecules.
        molecule.cluster_weight *=
          static_cast<double>(n_molecules_per_vertex[global_vertex_index])
          /
          static_cast<double>(n_cluster_molecules_per_vertex[global_vertex_index]);
      }

    return cell_energy_molecules;
  }



#define SINGLE_WEIGHTS_BY_VERTEX_INSTANTIATION(DIM, ATOMICITY, SPACEDIM) \
  template class WeightsByVertex< DIM, ATOMICITY, SPACEDIM >;            \
   
#define WEIGHTS_BY_VERTEX(R, X)                       \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,           \
              SINGLE_WEIGHTS_BY_VERTEX_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3)) X                \
   
  // WeightsByLumpedVertex class Instantiations.
  INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(WEIGHTS_BY_VERTEX)

#undef SINGLE_WEIGHTS_BY_VERTEX_INSTANTIATION
#undef WEIGHTS_BY_VERTEX


} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE
