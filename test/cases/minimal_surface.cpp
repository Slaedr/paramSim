#include <set>
#include <limits>
#include <gtest/gtest.h>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>

#include "../../cases/minimal_surface/minimal_surface.hpp"
#include "../../cases/minimal_surface/gaussians.hpp"

#include "../../utils/cmdparser.hpp"

using namespace paramsim;
namespace bpo = boost::program_options;

constexpr int dim = 2;

class MinimalSurfaceDiskSinusoidal : public testing::Test
{
protected:
    MinimalSurfaceDiskSinusoidal() : fe(1)
    {
        bpo::options_description common_desc
            ("Solves one problem given one set of parameters.");
        const bpo::variables_map common_cmdmap = get_cmd_args(0, NULL, common_desc);
        msds.initialize(common_cmdmap);
    }

    static constexpr double eps = std::numeric_limits<double>::epsilon();
    dealii::FE_Q<dim> fe;
    cases::MinSurfDiskSinusoidal<dim> msds;
};


TEST_F(MinimalSurfaceDiskSinusoidal, DefaultGeometryIsUnitBall)
{
    auto geom = msds.get_geometry();
    ASSERT_TRUE(std::dynamic_pointer_cast<const paramsim::geom::Ball<2>>(geom));
    dealii::Triangulation<2> tria;

    geom->generate_grid(tria, 4);

    for (auto ifa : tria.active_face_iterators()) {
        if(ifa->at_boundary()) {
            for(unsigned iv = 0; iv < ifa->n_vertices(); iv++) {
                const auto point = ifa->vertex(iv);
                ASSERT_TRUE(std::abs(point.norm() - 1.0) < eps);
            }
        }
    }
}

TEST_F(MinimalSurfaceDiskSinusoidal, DefaultBoundaryTags)
{
    auto geom = msds.get_geometry();
    ASSERT_TRUE(std::dynamic_pointer_cast<const paramsim::geom::Ball<2>>(geom));
    dealii::Triangulation<2> tria;
    geom->generate_grid(tria, 1);
    geom->set_boundary_ids(tria);
    dealii::DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    std::set<dealii::types::boundary_id> bids{1};

    auto n_bc_dofs = dof_handler.n_boundary_dofs(bids);

    EXPECT_EQ(n_bc_dofs, 4);
}


class MinimalSurfaceCubeSinusoidal : public testing::Test
{
protected:
    MinimalSurfaceCubeSinusoidal() : fe(1)
    {
        bpo::options_description common_desc
            ("Solves one problem given one set of parameters.");
        const bpo::variables_map common_cmdmap = get_cmd_args(0, NULL, common_desc);
        msds.initialize(common_cmdmap);
    }

    static constexpr double eps = std::numeric_limits<double>::epsilon();
    dealii::FE_Q<dim> fe;
    cases::MinSurfCubeSinusoidal<dim> msds;
};


TEST_F(MinimalSurfaceCubeSinusoidal, DefaultGeometryIsUnitCube)
{
    auto geom = msds.get_geometry();

    ASSERT_TRUE(std::dynamic_pointer_cast<const paramsim::geom::Cube<2>>(geom));
}

TEST_F(MinimalSurfaceCubeSinusoidal, BoundaryTags)
{
    auto geom = msds.get_geometry();
    dealii::Triangulation<2> tria;
    geom->generate_grid(tria, 3);
    geom->set_boundary_ids(tria);
    dealii::DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    constexpr double coord_tol = 1e-10;

    for(const auto &cell : dof_handler.active_cell_iterators()) {
        for(const auto &face : cell->face_iterators()) {
            if(!face->at_boundary()) {
                continue;
            }
            const auto n_verts = face->n_vertices();
            bool on_left_boundary = true;
            for(auto i = 0u; i < n_verts; i++) {
                const auto point = face->vertex(i);
                if(std::abs(point[0] + 1.0) > coord_tol) {
                    on_left_boundary = false;
                }
            }

            if(on_left_boundary) {
                EXPECT_TRUE(face->boundary_id() == 1);
            } else {
                EXPECT_TRUE(face->boundary_id() == 2);
            }
        }
    }
}


class MinimalSurfaceCubePolynomial : public testing::Test
{
protected:
    MinimalSurfaceCubePolynomial()
        : params_({{2.0, 3.2, -1.3, 0.4}}, -1.5), dbc_(params_)
    {
    }

    cases::minsurf_poly::Params<2> params_;
    cases::minsurf_poly::Dirichlet<2> dbc_;
};

TEST_F(MinimalSurfaceCubePolynomial, DirichletConditionIsZeroAtBoundaries)
{
    dealii::Point<2> pbottom{0.2, -1.0};
    dealii::Point<2> ptop{0.1, 1.0};
    dealii::Point<2> pin{-0.2, 0.5};

    EXPECT_NEAR(dbc_.value(pbottom), 0.0, 1e-14);
    EXPECT_NEAR(dbc_.value(ptop), 0.0, 1e-15);
}

TEST_F(MinimalSurfaceCubePolynomial, DirichletConditionIsZeroAtBoundaries2)
{
    cases::minsurf_poly::Params<2> params({{-1.1, -2.213, 1.9, -0.003}}, -1.0/3);
    cases::minsurf_poly::Dirichlet<2> dbc(params);
    dealii::Point<2> pbottom{0.2, -1.0};
    dealii::Point<2> ptop{0.1, 1.0};

    EXPECT_NEAR(dbc.value(pbottom), 0.0, 1e-15);
    EXPECT_NEAR(dbc.value(ptop), 0.0, 1e-15);
}

TEST_F(MinimalSurfaceCubePolynomial, DirichletConditionHasKnownValue)
{
    dealii::Point<2> pin{-0.2, 0.5};

    EXPECT_DOUBLE_EQ(dbc_.value(pin), -12.693750000000001);
}

TEST(MinimalSurfaceCubeGaussians, DefaultDirichletConditionIsZeroAtBoundaries)
{
    cases::minsurf_gauss::Dirichlet<2> dbc;
    dealii::Point<2> pbottom{0.2, -1.0};
    dealii::Point<2> ptop{0.1, 1.0};

    EXPECT_NEAR(dbc.value(pbottom), 0.0, 1e-15);
    EXPECT_NEAR(dbc.value(ptop), 0.0, 1e-15);
}

TEST(MinimalSurfaceCubeGaussians, DirichletConditionIsZeroAtBoundaries)
{
    cases::minsurf_gauss::Params<2> params({{-0.7, -0.1, 0.9}}, {{0.3,-0.8,0.6}}, 1.0/3);
    cases::minsurf_gauss::Dirichlet<2> dbc(params);
    dealii::Point<2> pbottom{0.2, -1.0};
    dealii::Point<2> ptop{0.1, 1.0};

    EXPECT_NEAR(dbc.value(pbottom), 0.0, 1e-14);
    EXPECT_NEAR(dbc.value(ptop), 0.0, 1e-15);
}
