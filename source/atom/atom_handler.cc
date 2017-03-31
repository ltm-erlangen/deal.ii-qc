

#include <deal.II/numerics/data_out.h>

#include <dealiiqc/atom/atom_handler.h>


namespace dealiiqc
{

  template<int dim>
  AtomHandler<dim>::AtomHandler( const ConfigureQC &configure_qc)
    :
    configure_qc(configure_qc)
  {
  }

  template<int dim>
  bool AtomHandler<dim>::is_energy_atom( const Atom<dim> &a)
  {
    // TODO: check that the atoms.parent_cell is not NULL
    //       and that it is a valid actice_cell
    // TODO:? additionally check if the atom actually belongs
    //        to the parent_cell
    for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
      if (  (a.parent_cell->vertex(v)- a.position).norm_square()
            < dealii::Utilities::fixed_power<2>( configure_qc.get_max_search_radius()) )
        return true;
    return false;
  }

  template<int dim>
  bool AtomHandler<dim>::is_cluster_atom( const Atom<dim> &a)
  {
    // TODO: check that the atoms.parent_cell is not NULL
    //       and that it is a valid actice_cell
    // TODO:? additionally check if the atom actually belongs
    //        to the parent_cell
    for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
      if (  (a.parent_cell->vertex(v)- a.position).norm_square()
            < dealii::Utilities::fixed_power<2>( configure_qc.get_cluster_radius()) )
        return true;
    return false;
  }

  template<int dim>
  void AtomHandler<dim>::parse_atoms_and_assign_to_cells( const MeshType &mesh,
                                                          const MPI_Comm &comm)
  {
    // TODO: Move to list_atoms?
    std::vector<Atom<dim>> vector_atoms;
    ParseAtomData<dim> atom_parser;

    if (!(configure_qc.get_atom_data_file()).empty() )
      {
        const std::string atom_data_file = configure_qc.get_atom_data_file();
        std::fstream fin(atom_data_file, std::fstream::in );
        atom_parser.parse( fin, vector_atoms, charges, masses);
      }
    else if ( !(* configure_qc.get_stream()).eof() )
      {
        atom_parser.parse( *configure_qc.get_stream(), vector_atoms, charges, masses);
      }
    else
      AssertThrow( false,
                   ExcMessage("Atom data was not provided neither as an auxiliary "
                              "data file nor at the end of the parameter file!"));

    // Construct a vector of bools that correspond
    // to vertices of locally owned cells
    std::vector<bool> locally_active_vertices( mesh.get_triangulation().n_vertices(),
                                               false);

    for ( typename MeshType::active_cell_iterator
          cell = mesh.begin_active();
          cell != mesh.end(); ++cell)
      {
        // Mark (true) all the vertices of the locally owned cell
        if ( cell->is_locally_owned () )
          for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
            locally_active_vertices[cell->vertex_index(v)] = true;
      }

    std::vector<typename MeshType::active_cell_iterator> ghost_cells;

    // If the number of MPI processes is just one,
    // there are no ghost cells. All active cells
    // are relevant cells.
    if ( dealii::Utilities::MPI::n_mpi_processes(comm) !=1)
      {
        ghost_cells = GridTools::compute_ghost_cell_layer_within_distance( mesh,
                      configure_qc.get_max_search_radius());
        for ( auto cell : ghost_cells)
          for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
            locally_active_vertices[cell->vertex_index(v)] = true;
      }

    typename std::vector<Atom<dim>> thrown_atoms;

    bool found_cell = false;
    for ( auto atom : vector_atoms )
      {
        found_cell = false;
        try
          {
            std::pair<typename MeshType::active_cell_iterator, Point<dim> >
            my_pair = GridTools::find_active_cell_around_point( MappingQ1<dim>(),
                                                                mesh,
                                                                atom.position,
                                                                locally_active_vertices);

            atom.reference_position = GeometryInfo<dim>::project_to_unit_cell(my_pair.second);
            atom.parent_cell = my_pair.first;
            found_cell = true;
            atoms.insert( std::make_pair( my_pair.first, atom ));
          }
        catch ( dealii::GridTools::ExcPointNotFound<dim> &)
          {
            // The atom is outside the cells that are relevant
            // to this MPI process. Ensuring quiet execution.
          }

        if ( !found_cell )
          thrown_atoms.push_back(atom);
      }

    Assert( atoms.size()+thrown_atoms.size()==vector_atoms.size(),
            ExcInternalError());
  }


  template<int dim>
  void AtomHandler<dim>::update_energy_atoms()
  {
    // Check each atom in atoms
    // if it's associated to a locally owned cell
    // if it is then check if it's an energy atom
    // if it is then check if it's a cluster atom
    for ( const auto &atom : atoms )
      if ( atom.first->is_locally_owned())
        if ( is_energy_atom( atom.second))
          {
            CellAtomsIterator cell_atoms_iterator = energy_atoms.insert(atom);
            if ( is_cluster_atom( atom.second))
              cluster_atoms_iterator.insert( std::make_pair( atom.first, cell_atoms_iterator) );
          }
  }


  template<int dim>
  void AtomHandler<dim>::write_cell_data( const MeshType &mesh,
                                          std::ostream &out)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler (mesh);
    unsigned int n_active_cells = mesh.get_tria().n_active_cells();

    // set the size of cell data vectors to the number of active cells
    Vector<float> n_atoms_per_cell,
           n_energy_atoms_per_cell,
           n_cluster_atoms_per_cell;

    n_atoms_per_cell.reinit ( n_active_cells);
    n_energy_atoms_per_cell.reinit ( n_active_cells);
    n_cluster_atoms_per_cell.reinit ( n_active_cells);

    for ( typename MeshType::active_cell_iterator
          cell  = mesh.begin_active();
          cell != mesh.end(); ++cell)
      {
        n_atoms_per_cell        [cell->active_cell_index()] = static_cast<float>(atoms.count(cell));
        n_energy_atoms_per_cell [cell->active_cell_index()] = static_cast<float>(energy_atoms.count(cell));
        n_cluster_atoms_per_cell[cell->active_cell_index()] = static_cast<float>(cluster_atoms_iterator.count(cell));
      }
    data_out.attach_dof_handler (mesh);

    data_out.add_data_vector ( n_atoms_per_cell, "n_atoms");
    data_out.add_data_vector ( n_energy_atoms_per_cell, "n_energy_atoms");
    data_out.add_data_vector ( n_cluster_atoms_per_cell, "n_cluster_atoms");

    data_out.build_patches ();
    data_out.write_vtk(out);
    AssertThrow ( out, ExcIO());
    //data_out.clear();
  }

  template class AtomHandler<1>;
  template class AtomHandler<2>;
  template class AtomHandler<3>;

} // dealiiqc namespace
