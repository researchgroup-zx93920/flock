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

std::ostream& operator << (std::ostream& o, const vertexPin& x) {
    o << "********\nFrom : "<<x.from<<"\nVia : "<<x.via<<"\nTo : "<<x.to<<"\nSkip : "<<x.skip;
    std::cout<<"\nAccumualated : "<<x.recirculation<<"\n********"<<std::endl;
    return o;
}