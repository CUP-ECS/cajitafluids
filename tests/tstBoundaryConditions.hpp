#ifndef _TST_BOUNDARYCONDITIONS_HPP_
#define _TST_BOUNDARYCONDITIONS_HPP_

#include "gtest/gtest.h"

// Include Statements

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <BoundaryConditions.hpp>

#include "tstDriver.hpp"
#include "tstProblemManager.hpp"

template <class T>
class BoundaryConditionsTest : public ProblemManagerTest<T>
{
  protected:
    void SetUp() { ProblemManagerTest<T>::SetUp(); }

    void TearDown() { ProblemManagerTest<T>::TearDown(); }
};

#endif // _TST_BOUNDARYCONDITIONS_HPP_
