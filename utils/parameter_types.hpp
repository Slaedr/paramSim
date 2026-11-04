#ifndef PARAMSIM_UTILS_PARAMETER_TYPES_HPP_
#define PARAMSIM_UTILS_PARAMETER_TYPES_HPP_

namespace paramsim {

/** Selects which solution is written on retained coarse meshes.
 *
 * @note Projection is reserved for a future transfer implementation.
 */
enum class CoarseGridSolution {
    /** Write each selected mesh's own converged simulation result. */
    native_simulation,
    /** Interpolate the converged finest solution onto selected meshes. */
    interpolation,
    /** Reserved for future projection of the finest solution. */
    projection
};

} // namespace paramsim

#endif // PARAMSIM_UTILS_PARAMETER_TYPES_HPP_
