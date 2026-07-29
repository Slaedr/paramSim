#ifndef PARAMSIM_CASES_CUBE_CUBE_CASE_HPP_
#define PARAMSIM_CASES_CUBE_CUBE_CASE_HPP_

#include <memory>

#include <deal.II/base/function.h>

#include "../case.hpp"

namespace paramsim {
namespace cases {
namespace cube {

/**
 * @brief Shared setup for cube cases whose Dirichlet profile applies to
 * every face of the cube boundary.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
class CubeCase : public Case<dim> {
protected:
    /**
     * @brief Sets the cube geometry and a single Dirichlet region covering
     * the entire boundary, evaluated with `profile`.
     *
     * @param profile Boundary profile to apply on every face.
     */
    void initialize_dirichlet_everywhere(
        std::shared_ptr<dealii::Function<dim>> profile);
};

} // namespace cube
} // namespace cases
} // namespace paramsim

#endif
