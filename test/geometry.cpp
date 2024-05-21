#include <limits>
#include <gtest/gtest.h>

#include <deal.II/grid/tria.h>

#include "../geometrybase.hpp"

namespace ps = paramsim;

constexpr int dim = 2;

class GeometryCube2d: public testing::Test
{
protected:
    GeometryCube2d()
    {
        std::vector<typename ps::DomainGeometry<dim>::bc_mark_desc> bcmarks;
        bcmarks.push_back(std::make_pair(bcid0,
            [](const dealii::Point<dim>& p) {
            if(std::abs(p[0] - (-1.0)) > eps) {
                return true;
            } else {
                return false;
            }
            }));
        bcmarks.push_back(std::make_pair(bcid1,
            [](const dealii::Point<dim>& p) {
            if(std::abs(p[0] - (-1.0)) <= eps) {
                return true;
            } else {
                return false;
            }
            }));
        geom = std::make_shared<ps::geom::Cube<dim>>(bcmarks);
    }

    const unsigned bcid0 = 3;
    const unsigned bcid1 = 5;
    std::shared_ptr<ps::DomainGeometry<dim>> geom;
    static constexpr double eps = 1000*std::numeric_limits<double>::epsilon();
};

TEST_F(GeometryCube2d, SetsBoundaryIDsCorrectly)
{
    dealii::Triangulation<2> tria;
    geom->generate_grid(tria, 3);

    geom->set_boundary_ids(tria);

    for(const auto &cell : tria.cell_iterators()) {
        for(const auto &face : cell->face_iterators()) {
            if(!face->at_boundary()) {
                continue;
            }
            const auto n_verts = face->n_vertices();
            bool on_left_boundary = true;
            for(auto i = 0u; i < n_verts; i++) {
                const auto point = face->vertex(i);
                if(std::abs(point[0] + 1.0) > eps) {
                    on_left_boundary = false;
                }
            }
            if(on_left_boundary) {
                EXPECT_TRUE(face->boundary_id() == bcid1);
            } else {
                EXPECT_TRUE(face->boundary_id() == bcid0);
            }
        }
    }
}
