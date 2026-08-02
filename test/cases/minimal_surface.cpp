#include <set>
#include <limits>
#include <gtest/gtest.h>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>

#include "../../cases/minimal_surface/minimal_surface.hpp"

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

