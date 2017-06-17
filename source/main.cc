

#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II/base/parameter_handler.h>

#include <deal.II-qc/core/qc.h>
#include <deal.II-qc/version.h>

// instantiations:
# include <boost/preprocessor/facilities/empty.hpp>
# include <boost/preprocessor/list/at.hpp>
# include <boost/preprocessor/list/for_each_product.hpp>
# include <boost/preprocessor/tuple/elem.hpp>
# include <boost/preprocessor/tuple/to_list.hpp>

// add supported dimensions
#define QC_DIM BOOST_PP_TUPLE_TO_LIST(3,(1,2,3))

// List of potentials
#define QC_POT \
  BOOST_PP_TUPLE_TO_LIST(\
                         2,\
                         ( \
                           ( Potential::PairCoulWolfManager, "Coulomb Wolf"), \
                           ( Potential::PairLJCutManager, "LJ") \
                         )\
                        )\
   
// Accessors for potentials
# define QC_POT_CLASS(TS) BOOST_PP_TUPLE_ELEM(2, 0, TS)
# define QC_POT_NAME(TS)  BOOST_PP_TUPLE_ELEM(2, 1, TS)

// Accessors for dimension and potential-string pairs
# define QC_GDIM(DP) BOOST_PP_TUPLE_ELEM(2, 0, DP)
# define QC_GTS(DP)  BOOST_PP_TUPLE_ELEM(2, 1, DP)

#define DOIF(R, DP) \
  else if ( (dim == QC_GDIM(DP)) && (pot.compare(QC_POT_NAME(QC_GTS(DP))) == 0) ) \
    { \
      QC<QC_GDIM(DP), QC_POT_CLASS(QC_GTS(DP))> problem(config); \
      problem.run (); \
    } \
   

int main (int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace dealiiqc;

      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
          dealii::numbers::invalid_unsigned_int);
      // if no input provided
      AssertThrow(argc > 1,ExcMessage("Parameter file is required as an input argument"));

      std::string parameter_filename = argv[1];
      {
        std::ifstream ifs(parameter_filename);
        AssertThrow( ifs, ExcIO() );
      }
      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::ifstream>( parameter_filename );

      ConfigureQC config(prm_stream);
      const unsigned int dim = config.get_dimension();
      const std::string pot = config.get_pair_potential_type();

      if (dim > 3)
        {
          Assert(false, ExcNotImplemented());
        }
      //BOOST_PP_LIST_FOR_EACH_PRODUCT(DOIF, 2, (QC_DIM,QC_POT))

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

      return 1;
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
      return 1;
    }

  return 0;
}
