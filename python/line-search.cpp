#include "line-search.hpp"

namespace pymannumopt {
void exposeLineSearch(py::module_ main)
{
  using namespace lineSearch;

  auto m = main.def_submodule("lineSearch");
  py::class_<Base>(m, "Base");

  py::class_<Armijo, Base>(m, "Armijo")
    .def(py::init<>())
    ;

  py::class_<BisectionWeakWolfe, Base>(m, "BisectionWeakWolfe")
    .def(py::init<>())
    ;
}
} // namespace pymannumopt
