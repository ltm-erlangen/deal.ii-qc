
#include <dealiiqc/io/parse_atom_data.h>

namespace dealiiqc
{
  using namespace dealii;

  template<int dim>
  ParseAtomData<dim>::ParseAtomData()
  :
  n_atoms(0),n_atom_types(0), line_no(0)
  {}

  template<int dim>
  void ParseAtomData<dim>::parse( std::istream & is, std::vector<Atom<dim>>& atoms)
  {
    std::string line;

    // Some temporary variables
    unsigned int nbonds = 0;
    unsigned int nangles = 0;
    unsigned int ndihedrals = 0;
    unsigned int nimpropers = 0;

    // Dimensions of the simulation box
    double xlo(0.),xhi(0.),
	   ylo(0.),yhi(0.),
	   zlo(0.),zhi(0.);

    // Tilts of the simulation box
    double xy(0.),xz(0.),yz(0.);

    // Read comment line (first line)
    line_no++;
    if(!std::getline(is,line))
      Assert( false, ExcReadFailed(line_no));

    // Read main header declarations
    while( true )
      {
	// Skip empty lines
	if( skip_read(is,line) )
	  Assert( false, ExcReadFailed(line_no));

	// Read and store n_atoms, nbonds, ...
	if(line.find("atoms") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%u", &n_atoms) != 1)
	    Assert( false, ExcInvalidValue(line_no,"bonds"));
	  }
	else if(line.find("bonds") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%u", &nbonds) != 1)
	      Assert( false, ExcInvalidValue(line_no,"bonds"));
	    Assert( nbonds==0, ExcIrrelevant(line_no));
	  }
	else if(line.find("angles") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%u", &nangles) != 1)
	      Assert( false, ExcInvalidValue(line_no,"angles"));
	    Assert( nangles==0, ExcIrrelevant(line_no));
	  }
	else if(line.find("dihedrals") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%u", &ndihedrals) != 1)
	      Assert( false, ExcInvalidValue(line_no,"dihedrals"));
	    Assert( ndihedrals==0, ExcIrrelevant(line_no));
	  }
	else if(line.find("impropers") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%u", &nimpropers) != 1)
	      Assert( false, ExcInvalidValue(line_no,"impropers"));
	    Assert( nimpropers==0, ExcIrrelevant(line_no));
	  }
	else if(line.find("atom types") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%u", &n_atom_types) != 1)
	      Assert( false , ExcInvalidValue(line_no,"number of atom types"));
	  }
	else if(line.find("xlo xhi") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%lf %lf", &xlo, &xhi) != 2)
	      Assert( false, ExcInvalidValue(line_no,"simulation box dimensions"));
	  }
	else if(line.find("ylo yhi") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%lf %lf", &ylo, &yhi) != 2)
	    Assert( false, ExcInvalidValue(line_no,"simulation box dimensions"));
	  }
	else if(line.find("zlo zhi") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%lf %lf", &zlo, &zhi) != 2)
	    Assert( false, ExcInvalidValue(line_no,"simulation box dimensions"));
	  }
	else if(line.find("xy xz yz") != std::string::npos)
	  {
	    if(sscanf(line.c_str(), "%lf %lf %lf", &xy, &xz, &yz) != 3)
	      Assert( false, ExcInvalidValue(line_no,"simulation box tilts"));
	    Assert(xy==xz==yz==0, ExcIrrelevant(line_no));
	  }
	else break;
      }

    // resize atoms with new number of atoms information
    atoms.resize(n_atoms);

    Assert( xhi >= xlo || yhi >= ylo || zhi >= zlo, ExcMessage("Invalid simulation box size"))
    // TODO: Extra Asserts with dim and simulation box dimensions
    // TODO: copy simulation box dimensions, check for inconsistent mesh?

    // Read sections with keyword headings (Atoms, Masses)
    // For now ignoring other keyword headings // TODO?
    while( true )
      {
	if( begins_with<std::string>(line, "Masses") )
	  {
	    if( parse_masses(is) )
	      break;
	    else if( skip_read(is,line) )
	      break;
	    else
	      continue;
	  }
	else if(begins_with<std::string>(line,"Atoms"))
	  {
	    if( parse_atoms(is, atoms))
	      break;
	    else if( skip_read(is,line) )
	      break;
	    else
	      continue;
	  }
	else
	  break;
      }
  }

  template< int dim>
  bool ParseAtomData<dim>::parse_atoms( std::istream & is,
					std::vector<Atom<dim>>& atoms )
  {
    bool end = false;
    std::string line;
    size_t atom_index, tmp1;
    unsigned int tmp2;
    double tmp_q, tmp_x, tmp_y, tmp_z;
    // TODO: atom type
    std::map<unsigned int,size_t> t;

    // Store atom attributes and positions
    for(unsigned int i=1; i<=n_atoms; ++i)
      {
	// Skip read each line and check its not end of stream
	// Important to have all the atom attributes
	if( skip_read(is,line) )
	  Assert( false, ExcMessage("Reached end of stream "
				    "Important atom attributes missing"));
	if(sscanf(line.c_str(),
		  "%zu %zu %u %lf %lf %lf %lf",
		  &atom_index, &tmp1, &tmp2, &tmp_q, &tmp_x, &tmp_y, &tmp_z ) !=7)
	  Assert( false, ExcMessage("Invalid atom values"));
	t[tmp2]          = atom_index;
	atoms[atom_index-1].q    = tmp_q;
	// TODO: 1 and 2 dim cases could be potentially incorrect
	// Possible way to correct this is to ask user axes dimensionality
	if(dim==1)
	  atoms[atom_index-1].position[0] = tmp_x;
	else if(dim==2)
	  {
	    atoms[atom_index-1].position[0] = tmp_x;
	    atoms[atom_index-1].position[1] = tmp_y;
	  }
	else if(dim==3)
	  {
	    atoms[atom_index-1].position[0] = tmp_x;
	    atoms[atom_index-1].position[1] = tmp_y;
	    atoms[atom_index-1].position[2] = tmp_z;
	  }
      }
    if( is.eof() )
      end = true;

    return end;
  }

  template< int dim>
  bool ParseAtomData<dim>::parse_masses( std::istream & is)
  {
    bool end = false;
    std::string line;
    size_t type;
    double tmp_mass;
    std::vector<double> masses(n_atom_types, 0.);

    // Store atom masses for different atom types
    for(unsigned int i=1; i<=n_atom_types; ++i)
      {
	// Skip read each line and check its not end of stream
	// Important to have all the atom attributes
	if( skip_read(is,line) )
	  Assert( false, ExcReadFailed(line_no));
	// TODO: Use masses of different types of atom for FIRE minimization scheme?
	if(sscanf(line.c_str(), "%zu %lf", &type, &tmp_mass) !=2)
	  Assert( false, ExcInvalidValue(line_no,"Mass"));
	masses[type] = tmp_mass;
      }

    if( is.eof() )
      end = true;

    return end;
  }

  template<int dim>
  bool ParseAtomData<dim>::skip_read(std::istream & is, std::string& line)
  {
    bool end = false;
    while( true )
      {
	// Check end of stream, if it is set end to false and exit loop
	if(is.eof())
	  {
	    end = true;
	    break;
	  }

	++line_no;
	// Read new line, if cannot read throw exception
	if(!std::getline(is,line))
	  Assert( false, ExcReadFailed(line_no));

	// Ignore empty lines
	if( line.find_first_not_of(" \t\n\r") == std::string::npos)
	  continue;
	else
	  break;
      }

    // Find if # is present
    size_t trim_pos = line.find("#");

    // Trim all the stuff after #
    if( trim_pos != std::string::npos )
      line.resize( trim_pos );

    // Trim all line with " \t\n\r\f\v"
    trim(line);

    return end;
  }

  template      ParseAtomData<1>::ParseAtomData();
  template      ParseAtomData<2>::ParseAtomData();
  template      ParseAtomData<3>::ParseAtomData();
  template bool ParseAtomData<1>::parse_masses(std::istream &);
  template bool ParseAtomData<2>::parse_masses(std::istream &);
  template bool ParseAtomData<3>::parse_masses(std::istream &);
  template bool ParseAtomData<1>::parse_atoms(std::istream &, std::vector<Atom<1>>&);
  template bool ParseAtomData<2>::parse_atoms(std::istream &, std::vector<Atom<2>>&);
  template bool ParseAtomData<3>::parse_atoms(std::istream &, std::vector<Atom<3>>&);
  template void ParseAtomData<1>::parse(std::istream &, std::vector<Atom<1>>&);
  template void ParseAtomData<2>::parse(std::istream &, std::vector<Atom<2>>&);
  template void ParseAtomData<3>::parse(std::istream &, std::vector<Atom<3>>&);
  template bool ParseAtomData<1>::skip_read(std::istream &, std::string &);
  template bool ParseAtomData<2>::skip_read(std::istream &, std::string &);
  template bool ParseAtomData<3>::skip_read(std::istream &, std::string &);

} /* namespace dealiiqc */
