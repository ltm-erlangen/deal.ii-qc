
#include <deal.II/grid/grid_tools.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/atom/parse_atom_data.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace CellMoleculeTools
{

  template<int dim, int atomicity, int spacedim>
  std::pair
  <
  types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>
  ,
  unsigned int
  >
  molecules_range_in_cell
  (const types::CellIteratorType<dim, spacedim>                     &cell,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules)
  {
    AssertThrow (!cell_molecules.empty(),
                 ExcMessage("The given CellMoleculeContainer is empty!"));

    const types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>
    cell_molecule_range = cell_molecules.equal_range(cell);

    const types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
    &cell_molecule_range_begin = cell_molecule_range.first,
     &cell_molecule_range_end  = cell_molecule_range.second;

    if (cell_molecule_range_begin == cell_molecule_range_end)
      // Quickly return the following if cell is not
      // found in the CellMoleculeContainerType object
      return std::make_pair(std::make_pair(cell_molecule_range_begin,
                                           cell_molecule_range_end),
                            0);

    // Faster to get the number of molecules in the active cell by
    // computing the distance between first and second iterators
    // instead of calling count on cell_molecules.
    // Here we implicitly cast to unsigned int, but this should be OK as
    // we check that the result is the same as calling count()
    const unsigned int
    n_molecules_in_cell = std::distance (cell_molecule_range.first,
                                         cell_molecule_range.second);

    Assert (n_molecules_in_cell == cell_molecules.count(cell),
            ExcMessage("The number of molecules or energy molecules in the "
                       "cell counted using the distance between the iterator "
                       "ranges yields a different result than "
                       "cell_molecules.count(cell) or"
                       "cell_energy_molecules.count(cell)."));

    return std::make_pair(std::make_pair(cell_molecule_range_begin,
                                         cell_molecule_range_end),
                          n_molecules_in_cell);
  }



  template<int dim, int atomicity=1, int spacedim=dim>
  unsigned int
  n_cluster_molecules_in_cell
  (const types::CellIteratorType<dim, spacedim> &cell,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_energy_molecules)
  {
    const types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>
    cell_molecule_range = cell_energy_molecules.equal_range(cell);

    const types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
    &cell_molecule_range_begin = cell_molecule_range.first,
     &cell_molecule_range_end  = cell_molecule_range.second;

    if (cell_molecule_range_begin == cell_molecule_range_end)
      // Quickly return the following if cell is not
      // found in the CellMoleculeContainerType object
      return 0;

    unsigned int n_cluster_molecules_in_this_cell = 0;

    for (types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
         cell_molecule_iterator  = cell_molecule_range_begin;
         cell_molecule_iterator != cell_molecule_range_end;
         cell_molecule_iterator++)
      {
        Assert (cell_molecule_iterator->second.cluster_weight != numbers::invalid_cluster_weight,
                ExcMessage("At least one of the molecule's cluster weight is "
                           "not initialized to a valid number."
                           "This function should be called only after "
                           "setting up correct cluster weights."));
        if (cell_molecule_iterator->second.cluster_weight != 0)
          n_cluster_molecules_in_this_cell++;
      }

    return n_cluster_molecules_in_this_cell;
  }



  template<int dim, int atomicity, int spacedim>
  CellMoleculeData<dim, atomicity, spacedim>
  build_cell_molecule_data (std::istream                         &is,
                            const types::MeshType<dim, spacedim> &mesh,
                            double          ghost_cell_layer_thickness)
  {
    // TODO: Assign atoms to cells as we parse atom data ?
    //       relevant for when we have a large collection of atoms.
    std::vector<Molecule<spacedim, atomicity>> vector_molecules;
    ParseAtomData<spacedim, atomicity> atom_parser;

    // Prepare cell molecule data in this container.
    CellMoleculeData<dim, atomicity, spacedim> cell_molecule_data;

    cell_molecule_data.charges = NULL;
    std::vector<types::charge> charges;
    auto &masses         = cell_molecule_data.masses;
    auto &cell_molecules = cell_molecule_data.cell_molecules;

    if ( !is.eof() )
      atom_parser.parse (is, vector_molecules, charges, masses);
    else
      AssertThrow(false,
                  ExcMessage("The provided input stream is empty."));

    cell_molecule_data.charges =
      std::make_shared<std::vector<types::charge>>(charges);

    const unsigned int n_vertices =  mesh.get_triangulation().n_vertices();

    // In order to speed-up finding an active cell around atoms through
    // find_active_cell_around_point(), we will need to construct a
    // mask for vertices of locally owned cells and ghost cells
    std::vector<bool> locally_active_vertices( n_vertices,
                                               false);

    // Loop through all the locally owned cells and
    // mark (true) all the vertices of the locally owned cells.
    for (typename types::MeshType<dim, spacedim>::active_cell_iterator
         cell = mesh.begin_active();
         cell != mesh.end(); ++cell)
      if (cell->is_locally_owned())
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
          locally_active_vertices[cell->vertex_index(v)] = true;

    // This MPI process also needs to know certain active ghost cells
    // within a certain distance from locally owned cells.
    // This MPI process will also keep copy of atoms associated to
    // such active ghost cells.
    // ghost_cells vector will contain all such active ghost cells.
    // If the total number of MPI processes is just one,
    // the size of ghost_cells vector is zero.
    const
    std::vector<typename types::MeshType<dim, spacedim>::active_cell_iterator>
    ghost_cells =
      GridTools::
      compute_ghost_cell_layer_within_distance (mesh,
                                                ghost_cell_layer_thickness);

    // Loop through all the ghost cells computed above and
    // mark (true) all the vertices of the locally owned and active
    // ghost cells within ConfigureQC::ghost_cell_layer_thickness.
    for (auto cell : ghost_cells)
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        locally_active_vertices[cell->vertex_index(v)] = true;

    // TODO: If/when required collect all non-relevant atoms
    // (those that are not within a ghost_cell_layer_thickness
    // for this MPI process energy computation)
    // For now just add the number of molecules being thrown.
    // TODO: Add another typedef for molecules index?
    types::global_atom_index n_thrown_molecules=0;

    for ( auto molecule : vector_molecules )
      {
        bool atom_associated_to_cell = false;
        try
          {
            // Find the locally active cell of the provided mesh which
            // surrounds the initial location of the molecule.
            std::pair<
            typename types::MeshType<dim, spacedim>::active_cell_iterator
            ,
            Point<dim>
            >
            my_pair =
              GridTools::
              find_active_cell_around_point (MappingQ1<dim,spacedim>(),
                                             mesh,
                                             molecule_initial_location(molecule),
                                             locally_active_vertices);

            // Since in locally_active_vertices all the vertices of
            // the ghost cells are marked true, find_active_cell_around_point
            // could take the liberty to find a cell that is not a ghost cell
            // of a current MPI process but has one of it's vertices marked
            // true.
            // In such a case, we need to throw the atom and
            // continue associating remaining atoms.
            if (!my_pair.first->is_locally_owned() &&
                (std::find (ghost_cells.begin(),
                            ghost_cells.end(),
                            my_pair.first)==ghost_cells.end()))
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

            cell_molecules.insert( std::make_pair( my_pair.first, molecule ));
            atom_associated_to_cell = true;
          }
        catch (dealii::GridTools::ExcPointNotFound<dim> &)
          {
            // The atom is outside the cells that are relevant
            // to this MPI process. Ensuring quiet execution.
          }

        if ( !atom_associated_to_cell )
          n_thrown_molecules++;
      }

    Assert (cell_molecules.size()+n_thrown_molecules==vector_molecules.size(),
            ExcInternalError());

    return cell_molecule_data;
  }



  template <int dim, int atomicity, int spacedim>
  IndexSet
  extract_locally_relevant_dofs
  (const dealii::DoFHandler<dim, spacedim>                          &dof_handler,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules)
  {
    // Prepare dof index set in this container.
    IndexSet dof_set = dof_handler.locally_owned_dofs();

    // Note: The logic here is similar to that of
    // DoFTools::extract_locally_relevant_dofs().
    std::vector<dealii::types::global_dof_index> dof_indices;
    std::vector<dealii::types::global_dof_index> dofs_on_ghosts;

    // Loop over unique locally relevant cells.
    for (types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
         unique_key  = cell_molecules.cbegin();
         unique_key != cell_molecules.cend();
         unique_key  = cell_molecules.upper_bound(unique_key->first))
      {
        const auto &cell = unique_key->first;

        dof_indices.resize(cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        for (unsigned int i=0; i<dof_indices.size(); ++i)
          if (!dof_set.is_element(dof_indices[i]))
            dofs_on_ghosts.push_back(dof_indices[i]);
      }

    // Sort, fill into index set and compress out duplicates.
    std::sort(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
    dof_set.add_indices (dofs_on_ghosts.begin(),
                         std::unique (dofs_on_ghosts.begin(),
                                      dofs_on_ghosts.end()));
    dof_set.compress();

    return dof_set;
  }


#define SINGLE_CELL_MOLECULE_TOOLS_INSTANTIATION(DIM, ATOMICITY, SPACEDIM) \
  \
  template                                                                 \
  std::pair<types::CellMoleculeConstIteratorRangeType<DIM, ATOMICITY, SPACEDIM>, unsigned int> \
  molecules_range_in_cell<DIM, ATOMICITY, SPACEDIM>                      \
  (const types::CellIteratorType<DIM, SPACEDIM>                     &,   \
   const types::CellMoleculeContainerType<DIM, ATOMICITY, SPACEDIM> &);  \
  \
  template                                                               \
  unsigned int                                                           \
  n_cluster_molecules_in_cell<DIM, ATOMICITY, SPACEDIM>                  \
  (const types::CellIteratorType<DIM, SPACEDIM>                     &,   \
   const types::CellMoleculeContainerType<DIM, ATOMICITY, SPACEDIM> &);  \
  \
  template                                                               \
  CellMoleculeData<DIM, ATOMICITY, SPACEDIM>                             \
  build_cell_molecule_data (std::istream                         &,      \
                            const types::MeshType<DIM, SPACEDIM> &,      \
                            double                               );      \
  \
  template                                                               \
  IndexSet                                                               \
  extract_locally_relevant_dofs<DIM, ATOMICITY, SPACEDIM>                \
  (const types::MeshType<DIM, SPACEDIM>                             &,   \
   const types::CellMoleculeContainerType<DIM, ATOMICITY, SPACEDIM> &);

#define CELL_MOLECULE_TOOLS(R, X)                       \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,             \
              SINGLE_CELL_MOLECULE_TOOLS_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3)) X

  // MoleculeHandler class Instantiations.
  INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(CELL_MOLECULE_TOOLS)

#undef SINGLE_CELL_MOLECULE_TOOLS_INSTANTIATION
#undef CELL_MOLECULE_TOOLS


} // namespace CellMoleculeTools


DEAL_II_QC_NAMESPACE_CLOSE

