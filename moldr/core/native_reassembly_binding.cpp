// Python adapter for the optional native candidate generator.
#include "native_reassembly.h"

#include <boost/python.hpp>
#include <boost/version.hpp>
#include <stdexcept>
#include <utility>

static_assert(BOOST_VERSION == 108500, "Build with Boost 1.85 headers.");

namespace bp = boost::python;

namespace {
bp::object generate(const RDKit::ROMol &base, const RDKit::ROMol &fragment) {
  std::optional<moldr_native::Result> result;
  try {
    result = moldr_native::generate(base, fragment);
  } catch (const std::invalid_argument &e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    bp::throw_error_already_set();
  }
  if (!result) {
    return bp::object();
  }
  bp::list candidates;
  using ToPythonMol = bp::manage_new_object::apply<RDKit::ROMol *>::type;
  for (auto &[smiles, mol] : result->candidates) {
    // Return an owning rdchem.Mol just like the Python direct-transfer path.
    auto *owned = new RDKit::ROMol(std::move(*mol));
    bp::object py_mol{bp::handle<>(ToPythonMol()(owned))};
    candidates.append(bp::make_tuple(smiles, py_mol));
  }
  const auto valid = result->candidates.size();
  bp::dict stats;
  stats["generated"] = result->generated;
  stats["raw_unique"] = result->raw_unique;
  stats["valid"] = valid;
  stats["duplicates"] = result->generated - valid - result->invalid;
  stats["invalid"] = result->invalid;
  return bp::make_tuple(candidates, stats);
}
}  // namespace

BOOST_PYTHON_MODULE(_native_reassembly) {
  bp::import("rdkit.Chem");
  bp::def("generate", generate, (bp::arg("base"), bp::arg("fragment")));
}
