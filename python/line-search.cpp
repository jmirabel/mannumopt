#include "line-search.hpp"

namespace pymannumopt {
void exposeLineSearch(py::module_ main)
{
  using namespace lineSearch;

  auto m = main.def_submodule("lineSearch");
  py::class_<Base>(m, "Base");

  py::class_<Armijo, Base>(m, "Armijo")
    .def(py::init<>())
    .def_readwrite("r", &Armijo::r)
    .def_readwrite("c", &Armijo::c)
    .def_readwrite("amin", &Armijo::amin)
    ;

  py::class_<BisectionWeakWolfe, Base>(m, "BisectionWeakWolfe")
    .def(py::init<>())
    ;
}
} // namespace pymannumopt
