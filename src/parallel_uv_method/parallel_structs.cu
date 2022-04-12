#include "parallel_structs.h"

std::ostream &operator<<(std::ostream &o, const MatrixCell &x)
{
    o << "Cost(" <<x.row<<", " << x.col <<") = " << x.cost;
    return o;
}

std::ostream &operator<<(std::ostream &o, const vogelDifference &x)
{
    o << x.diff << " | Index of Minimum : " << x.ileast_1 << " | Index of second Minimum : " << x.ileast_2;
    return o;
}

std::ostream& operator << (std::ostream& o, const Variable& x) {
    o << x.value;
    return o;
}
