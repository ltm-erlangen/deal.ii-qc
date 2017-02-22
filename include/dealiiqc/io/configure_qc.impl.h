
namespace dealiiqc
{
  using namespace dealii;

  template< int dim>
  ConfigureQC<dim>::ConfigureQC(  )
  :
  prm(), mesh_file(std::string()), do_refinement(false), n_cycles(0)
  {}

  template< int dim>
  ConfigureQC<dim>::ConfigureQC( const std::string &filename )
  :
  prm(), mesh_file(std::string()), do_refinement(false), n_cycles(0)
  {
    configure_qc( filename );
  }

  template<int dim>
  void ConfigureQC<dim>::get_mesh( std::string &filename)
  {
    filename = mesh_file;
  }

  template<int dim>
  std::string ConfigureQC<dim>::get_mesh()
  {
    if( ! mesh_file.empty() )
      return mesh_file;
    else
      return std::string();
  }

  template<int dim>
  void ConfigureQC<dim>::configure_qc( const std::string &filename )
  {
    deallog << std::endl << "Parsing qc input file " << filename << std::endl
	      << "for a " << dim << " dimensional simulation. " << std::endl;

    prm.enter_subsection ("Configure mesh");
    {
      prm.declare_entry("mesh file", "",
		      Patterns::Anything (),
		      "Name of the mesh file "
		      "with dealii compatible mesh files");
      prm.declare_entry("do initial global refinement", "false",
		      Patterns::Bool(),
		      "Specify whether a global refinement "
		      "should be performed "
		      "on the initial grid provided to qc");
      prm.declare_entry("number of refinement cycles", "0",
		      Patterns::Integer(),
		      "Number of global mesh refinement steps "
		      "applied to initial grid");
    }
    prm.leave_subsection ();

    /* // TODO: Read atom infomration from LAMMPS like atom data file
    prm.enter_subsection ("Configure atoms");
    {
      // N atoms
      // a atom types
      //
      // Atoms
      // Atom_ID Atom_Type Atom_Charge Atom_X Atom_Y Atom_Z
    }
    prm.leave_subsection ();
    */

    prm.read_input(filename);
    prm.enter_subsection("Configure mesh");
    {
      mesh_file     = prm.get("mesh file");
      do_refinement = prm.get_bool ("do initial global refinement");
      n_cycles      = prm.get_integer("number of refinement cycles");
    }
    prm.leave_subsection();

  }

  // instantiations:
  // TODO: move to insta.in

  /*
  template class ConfigureQC<1>;
  template class ConfigureQC<2>;
  template class ConfigureQC<3>;

  template ConfigureQC<1>::ConfigureQC();
  template ConfigureQC<2>::ConfigureQC();
  template ConfigureQC<3>::ConfigureQC();

  template ConfigureQC<1>::ConfigureQC( const std::string & );
  template ConfigureQC<2>::ConfigureQC( const std::string & );
  template ConfigureQC<3>::ConfigureQC( const std::string & );

  template void ConfigureQC<1>::configure_qc ( const std::string & );
  template void ConfigureQC<2>::configure_qc ( const std::string & );
  template void ConfigureQC<3>::configure_qc ( const std::string & );
  */

}
