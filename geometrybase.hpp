#ifndef PARAMSIM_GEOMETRYBASE_HPP_
#define PARAMSIM_GEOMETRYBASE_HPP_

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/tria.h>

namespace paramsim {

using bc_id_t = dealii::types::boundary_id;

/** Abstraction for generating a Deal II Triangulation from some kind of geometry description.
 *
 * Also provides a routine for setting boundary face IDs for boundary conditions.
 */
template <int dim> 
class DomainGeometry
{
public:
    /// Type describing a boundary marker and the condition for a point to be on that boundary
    using bc_mark_desc = std::pair<bc_id_t, std::function<bool(const dealii::Point<dim>&)>>;

    DomainGeometry() { }
    
    DomainGeometry(const std::vector<bc_mark_desc>& bc_marks)
        : bciddesc(bc_marks)
    { }

    virtual void generate_grid(dealii::Triangulation<dim>& tria,
            const unsigned int initial_resolution) const = 0;

    void set_bc_mark_desc(const std::vector<bc_mark_desc>& bc_marks) {
        bciddesc = bc_marks;
    }

    void set_boundary_ids(dealii::Triangulation<dim>& tria) const
    {
        if(bciddesc.empty()) {
            return;
        }
        for (const auto &cell : tria.cell_iterators()) {
          for (const auto &face : cell->face_iterators()) {
            if (face->at_boundary()) {
              for (auto bcid : bciddesc) {
                if (bcid.second(face->center())) {
                  face->set_boundary_id(bcid.first);
                }
              }
            }
          }
        }
    }
protected:
    std::vector<bc_mark_desc> bciddesc;
};

/// Abstract type for a function on a facet
template <int dim>
class FaceFunction
{
public:
  virtual double value_normal(const dealii::Point<dim>& p, const dealii::Tensor<1,dim>& normal,
          const unsigned int = 0) const = 0;
};

}

#endif
