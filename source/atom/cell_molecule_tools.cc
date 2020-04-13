
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/atom/parse_atom_data.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace CellMoleculeTools
{
  template <int dim, int atomicity, int spacedim>
  std::pair<types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>,
            unsigned int>
  molecules_range_in_cell(
    const types::CellIteratorType<dim, spacedim> &cell,
    const types::CellMoleculeContainerType<dim, atomicity, spacedim>
      &cell_molecules)
  {
    AssertThrow(!cell_molecules.empty(),
                ExcMessage("The given CellMoleculeContainer is empty!"));

    const types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>
      cell_molecule_range = cell_molecules.equal_range(cell);

    const types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
      &cell_molecule_range_begin = cell_molecule_range.first,
      &cell_molecule_range_end   = cell_molecule_range.second;

    if (cell_molecule_range_begin == cell_molecule_range_end)
      // Quickly return the following if cell is not
      // found in the CellMoleculeContainerType object
      return {{cell_molecule_range_begin, cell_molecule_range_end}, 0};

    // Faster to get the number of molecules in the active cell by
    // computing the distance between first and second iterators
    // instead of calling count on cell_molecules.
    // Here we implicitly cast to unsigned int, but this should be OK as
    // we check that the result is the same as calling count()
    const unsigned int n_molecules_in_cell =
      std::distance(cell_molecule_range.first, cell_molecule_range.second);

    Assert(n_molecules_in_cell == cell_molecules.count(cell),
           ExcMessage("The number of molecules or energy molecules in the "
                      "cell counted using the distance between the iterator "
                      "ranges yields a different result than "
                      "cell_molecules.count(cell) or"
                      "cell_energy_molecules.count(cell)."));

    return {{cell_molecule_range_begin, cell_molecule_range_end},
            n_molecules_in_cell};
  }



  template <int dim, int atomicity, int spacedim>
  unsigned int
  n_cluster_molecules_in_cell(
    const types::CellIteratorType<dim, spacedim> &cell,
    const types::CellMoleculeContainerType<dim, atomicity, spacedim>
      &cell_energy_molecules)
  {
    const types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>
      cell_molecule_range = cell_energy_molecules.equal_range(cell);

    const types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
      &cell_molecule_range_begin = cell_molecule_range.first,
      &cell_molecule_range_end   = cell_molecule_range.second;

    if (cell_molecule_range_begin == cell_molecule_range_end)
      // Quickly return the following if cell is not
      // found in the CellMoleculeContainerType object
      return 0;

    unsigned int n_cluster_molecules_in_this_cell = 0;

    for (types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
           cell_molecule_iterator = cell_molecule_range_begin;
         cell_molecule_iterator != cell_molecule_range_end;
         cell_molecule_iterator++)
      {
        Assert(cell_molecule_iterator->second.cluster_weight !=
                 numbers::invalid_cluster_weight,
               ExcMessage("At least one of the molecule's cluster weight is "
                          "not initialized to a valid number."
                          "This function should be called only after "
                          "setting up correct cluster weights."));
        if (cell_molecule_iterator->second.cluster_weight != 0)
          n_cluster_molecules_in_this_cell++;
      }

    return n_cluster_molecules_in_this_cell;
  }



  template <int dim, int atomicity, int spacedim>
  CellMoleculeData<dim, atomicity, spacedim>
  build_cell_molecule_data(std::istream &                         is,
                           const Triangulation<dim, spacedim> &   mesh,
                           const GridTools::Cache<dim, spacedim> &grid_cache)
  {
    // TODO: Assign atoms to cells as we parse atom data ?
    //       relevant for when we have a large collection of atoms.
    std::vector<Molecule<spacedim, atomicity>> vector_molecules;
    ParseAtomData<spacedim, atomicity>         atom_parser;

    // Prepare cell molecule data in this container.
    CellMoleculeData<dim, atomicity, spacedim> cell_molecule_data;


    auto &masses         = cell_molecule_data.masses;
    auto &bonds          = cell_molecule_data.bonds;
    auto &cell_molecules = cell_molecule_data.cell_molecules;

    cell_molecule_data.charges = NULL;
    std::vector<types::charge> charges;

    if (!is.eof())
      atom_parser.parse(is, vector_molecules, charges, masses, bonds);
    else
      AssertThrow(false, ExcMessage("The provided input stream is empty."));

    cell_molecule_data.charges =
      std::make_shared<std::vector<types::charge>>(charges);

    const unsigned int n_vertices = mesh.n_vertices();

    // In order to speed-up finding an active cell around atoms through
    // find_active_cell_around_point(), we will need to construct a
    // mask for vertices of locally owned cells and relevant ghost cells
    std::vector<bool> locally_active_vertices(n_vertices, false);

    // Loop through all the locally relevant cells and
    // mark (true) all the vertices of the locally relevant cells i.e.,
    // locally owned cells plus certain active ghost cells
    // within a certain distance from locally owned cells.
    // This MPI process will also keep copy of atoms associated to
    // such active ghost cells.
    for (types::CellIteratorType<dim, spacedim> cell = mesh.begin_active();
         cell != mesh.end();
         ++cell)
      if (!cell->is_artificial())
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          locally_active_vertices[cell->vertex_index(v)] = true;

    // TODO: If/when required collect all non-relevant atoms
    // (those that are not within a ghost_cell_layer_thickness
    // for this MPI process energy computation)
    // For now just add the number of molecules being thrown.
    types::global_molecule_index n_thrown_molecules = 0;

    std::function<bool(const types::CellIteratorType<dim, spacedim> &)>
      predicate = [](const types::CellIteratorType<dim, spacedim> &cell) {
        return !cell->is_artificial();
      };

    // Prepare a bounding box for the current process.
    const std::pair<Point<spacedim>, Point<spacedim>>
      this_process_bounding_box =
        dealii::GridTools::compute_bounding_box(mesh, predicate);

    // Prepare an initial guess (which here is the first locally relevant cell).
    types::CellIteratorType<dim, spacedim> cell_hint;
    for (types::CellIteratorType<dim, spacedim> cell = mesh.begin_active();
         cell != mesh.end();
         ++cell)
      if (!cell->is_artificial())
        {
          cell_hint = cell;
          break;
        }

    for (auto molecule : vector_molecules)
      {
        bool atom_associated_to_cell = false;

        const Point<spacedim> &molecule_location =
          molecule_initial_location(molecule);

        // If the molecule is outside the bounding box,
        // throw the molecule and continue associating remaining molecules.
        if (dealiiqc::Utilities::is_outside_bounding_box(
              this_process_bounding_box.first,
              this_process_bounding_box.second,
              molecule_location))
          {
            n_thrown_molecules++;
            continue;
          }

        try
          {
            // Find the locally active cell of the provided mesh which
            // surrounds the initial location of the molecule.
            std::pair<types::CellIteratorType<dim, spacedim>, Point<dim>>
              my_pair = GridTools::find_active_cell_around_point(
                grid_cache,
                molecule_location,
                cell_hint,
                locally_active_vertices);

            // Since in locally_active_vertices some of the vertices of
            // the artificial cells are marked true, the function
            // GridTools::find_active_cell_around_point()
            // could take the liberty to find an artificial cell of the current
            // MPI process that has one of it's vertices marked true.
            // In such a case, we need to throw the molecule and
            // continue associating remaining molecules.
            if (my_pair.first->is_artificial())
              {
                n_thrown_molecules++;
                continue;
              }

            Point<dim> reference_position_in_cell =
              GeometryInfo<dim>::project_to_unit_cell(my_pair.second);

            // Molecule is in spacedim-dimensional space but its location
            // in the reference cell is in dim-dimensional space.
            // Copy dim-dimensional Point into spacedim-dimensional Point.
            for (int d = 0; d < dim; ++d)
              molecule.position_inside_reference_cell[d] =
                reference_position_in_cell[d];

            for (int d = dim; d < spacedim; ++d)
              molecule.position_inside_reference_cell[d] =
                std::numeric_limits<double>::signaling_NaN();

            cell_molecules.insert(std::make_pair(my_pair.first, molecule));
            atom_associated_to_cell = true;

            // The next atom is most likely geometrically inside this cell.
            cell_hint = my_pair.first;
          }
        catch (dealii::GridTools::ExcPointNotFound<dim> &)
          {
            // The atom is outside the cells that are relevant
            // to this MPI process. Ensuring quiet execution.
          }

        if (!atom_associated_to_cell)
          n_thrown_molecules++;
      }
#if DEBUG
    Assert(cell_molecules.size() + n_thrown_molecules ==
             vector_molecules.size(),
           ExcInternalError());

    types::global_molecule_index n_molecules_imported = 0;

    for (types::CellIteratorType<dim, spacedim> cell = mesh.begin_active();
         cell != mesh.end();
         ++cell)
      if (cell->is_locally_owned())
        n_molecules_imported +=
          molecules_range_in_cell<dim, atomicity, spacedim>(cell,
                                                            cell_molecules)
            .second;

    const parallel::Triangulation<dim, spacedim> *const pmesh =
      dynamic_cast<const parallel::Triangulation<dim, spacedim> *>(&mesh);

    // Get a consistent MPI_Comm.
    const MPI_Comm &mpi_communicator =
      pmesh != nullptr ? pmesh->get_communicator() : MPI_COMM_SELF;

    n_molecules_imported =
      dealii::Utilities::MPI::sum(n_molecules_imported, mpi_communicator);

    Assert(n_molecules_imported == vector_molecules.size(),
           ExcMessage("Some of the molecules are not associated with any cell "
                      "of the given Triangulation. It is possible that these "
                      "molecules are located outside the given "
                      "Triangulation!"));
#endif

    return cell_molecule_data;
  }



  template <int dim, int atomicity = 1, int spacedim = dim>
  double
  compute_molecule_density(
    const Triangulation<dim, spacedim> &triangulation,
    const types::CellMoleculeContainerType<dim, atomicity, spacedim>
      &cell_molecules)
  {
    double                       tria_volume = 0.;
    types::global_molecule_index n_molecules = 0;

    const parallel::Triangulation<dim, spacedim> *const pmesh =
      dynamic_cast<const parallel::Triangulation<dim, spacedim> *>(
        &triangulation);

    // Get a consistent MPI_Comm.
    const MPI_Comm &mpi_communicator =
      pmesh != nullptr ? pmesh->get_communicator() : MPI_COMM_SELF;

    for (types::CellIteratorType<dim, spacedim> cell =
           triangulation.begin_active();
         cell != triangulation.end();
         ++cell)
      if (cell->is_locally_owned())
        {
          tria_volume += cell->measure();
          n_molecules +=
            CellMoleculeTools::
              molecules_range_in_cell<dim, atomicity, spacedim>(cell,
                                                                cell_molecules)
                .second;
        }

    tria_volume = dealii::Utilities::MPI::sum(tria_volume, mpi_communicator);
    n_molecules = dealii::Utilities::MPI::sum(n_molecules, mpi_communicator);

    return static_cast<double>(n_molecules) / tria_volume;
  }



#define SINGLE_CELL_MOLECULE_TOOLS_INSTANTIATION(_DIM, _ATOMICITY, _SPACE_DIM) \
                                                                               \
  template std::pair<                                                          \
    types::CellMoleculeConstIteratorRangeType<_DIM, _ATOMICITY, _SPACE_DIM>,   \
    unsigned int>                                                              \
  molecules_range_in_cell<_DIM, _ATOMICITY, _SPACE_DIM>(                       \
    const types::CellIteratorType<_DIM, _SPACE_DIM> &,                         \
    const types::CellMoleculeContainerType<_DIM, _ATOMICITY, _SPACE_DIM> &);   \
                                                                               \
  template unsigned int                                                        \
  n_cluster_molecules_in_cell<_DIM, _ATOMICITY, _SPACE_DIM>(                   \
    const types::CellIteratorType<_DIM, _SPACE_DIM> &,                         \
    const types::CellMoleculeContainerType<_DIM, _ATOMICITY, _SPACE_DIM> &);   \
                                                                               \
  template CellMoleculeData<_DIM, _ATOMICITY, _SPACE_DIM>                      \
  build_cell_molecule_data(std::istream &,                                     \
                           const Triangulation<_DIM, _SPACE_DIM> &,            \
                           const GridTools::Cache<_DIM, _SPACE_DIM> &);        \
                                                                               \
  template double compute_molecule_density(                                    \
    const Triangulation<_DIM, _SPACE_DIM> &,                                   \
    const types::CellMoleculeContainerType<_DIM, _ATOMICITY, _SPACE_DIM> &);

#define CELL_MOLECULE_TOOLS(R, X)                       \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,             \
              SINGLE_CELL_MOLECULE_TOOLS_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3))                    \
  X

  // MoleculeHandler class Instantiations.
  INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(CELL_MOLECULE_TOOLS)

#undef SINGLE_CELL_MOLECULE_TOOLS_INSTANTIATION
#undef CELL_MOLECULE_TOOLS


} // namespace CellMoleculeTools


DEAL_II_QC_NAMESPACE_CLOSE
