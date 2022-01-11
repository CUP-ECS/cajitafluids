#ifndef EXACLAMR_TYPES_HPP
#define EXACLAMR_TYPES_HPP

#include <type_traits>

namespace ExaCLAMR {

    template <class Scalar>
    struct RegularMesh {
        // Scalar type for mesh floating point operations.
        using scalar_type = Scalar;
    };

    // Non-uniform mesh tag.
    template <class Scalar>
    struct AMRMesh {
        // Scalar type for mesh floating point operations.
        using scalar_type = Scalar;
    };

    // Type checker.
    template <class T>
    struct isExaMeshType : public std::false_type {
    };

    template <class Scalar>
    struct isExaMeshType<RegularMesh<Scalar>> : public std::true_type {
    };

    template <class Scalar>
    struct isExaMeshType<const RegularMesh<Scalar>> : public std::true_type {
    };

    template <class Scalar>
    struct isExaMeshType<AMRMesh<Scalar>> : public std::true_type {
    };

    template <class Scalar>
    struct isExaMeshType<const AMRMesh<Scalar>> : public std::true_type {
    };

    // Uniform mesh checker.
    template <class T>
    struct isRegularMesh : public std::false_type {
    };

    template <class Scalar>
    struct isRegularMesh<RegularMesh<Scalar>> : public std::true_type {
    };

    template <class Scalar>
    struct isRegularMesh<const RegularMesh<Scalar>> : public std::true_type {
    };

    // Non-uniform mesh checker.
    template <class T>
    struct isAMRMesh : public std::false_type {
    };

    template <class Scalar>
    struct isAMRMesh<AMRMesh<Scalar>> : public std::true_type {
    };

    template <class Scalar>
    struct isAMRMesh<const AMRMesh<Scalar>> : public std::true_type {
    };

} // namespace ExaCLAMR

#endif