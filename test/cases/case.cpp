#include <gtest/gtest.h>

#include "../../cases/case.hpp"
#include "../../cases/minimal_surface/minimal_surface.hpp"

namespace ps = paramsim;

TEST(Cases, CanCreateDefaultMinSurfCubeSinusoidalCase)
{
    std::shared_ptr<ps::Case<2>> case1 = ps::create_case<2>("minimal_surface_cube_sinusoidal");

    EXPECT_TRUE(std::dynamic_pointer_cast<ps::cases::MinSurfCubeSinusoidal<2>>(case1));
}
