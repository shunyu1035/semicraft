#include <iostream>
#include <Eigen/Dense>

// int main() {
//     Eigen::Matrix3d m;
//     m << 1, 2, 3,
//          4, 5, 6,
//          7, 8, 9;

//     std::cout << "Matrix m:" << std::endl << m << std::endl;
//     return 0;
// }

using Eigen::MatrixXd;
 
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}