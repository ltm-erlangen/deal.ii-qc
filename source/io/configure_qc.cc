
#include <dealiiqc/io/configure_qc.h>

namespace dealiiqc
{
  using namespace dealii;

  template< int dim>
  ConfigureQC<dim>::ConfigureQC(  )
  :
  prm(), mesh_file(std::string()), n_initial_global_refinements(1)
  {}source/io/configure_qc.cc

  template< int dim>
  ConfigureQC<dim>::ConfigureQC( const std::istringstream &iss )
  :
  ConfigureQC<dim>()
  {
    configure_qc( iss );
  }

  template<int dim>
  std::string ConfigureQC<dim>::get_mesh_file()
  {
    return mesh_file;
  }

  template<int dim>
  unsigned int ConfigureQC<dim>::get_n_initial_global_refinements()
  {
    return n_initial_global_refinements;
  }

  template<int dim>
  void ConfigureQC<dim>::configure_qc( const std::istringstream &iss )
  {
    // TODO: Write intput file name to the screen
    //deallog << std::endl << "Parsing qc input file " << filename << std::endl
    //	      << "for a " << dim << " dimensional simulation. " << std::endl;

    prm.enter_subsection ("Configure mesh");
    {
      prm.declare_entry("Mesh file", "",
		      Patterns::Anything(),
		      "Name of the mesh file "
		      "with dealii compatible mesh files");
      prm.declare_entry("Number of initial global refinements", "1",
		      Patterns::Integer(0),
		      "Number of global mesh refinement cycles "
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
    std::stringstream ess (iss.str());
    std::string abs_path, filename;
    ess >> abs_path; ess >> filename;

    prm.read_input("/home/ken/Documents/Git-repos/deal.ii-qc/tests/io/mesh_01/qc.prm");
    prm.enter_subsection("Configure mesh");
    {
      mesh_file                    = prm.get("Mesh file");
      n_initial_global_refinements = prm.get_integer("Number of initial global refinements");
    }
    mesh_file.insert(0, abs_path);
    prm.leave_subsection();

  }

  //#include "configure_qc.inst"

}
