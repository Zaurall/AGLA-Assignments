#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdio>

#ifdef WIN32
#define GNUPLOT_NAME "C:\\gnuplot\\bin\\gnuplot -persist"
#endif

class ColumnVector {
private:
    std::vector<double> vector;

public:
    ColumnVector(int size) {
        vector.resize(size);
    }

    int size() {
        return vector.size();
    }

    void printDouble() {
        for (int i = 0; i < vector.size(); ++i) {
            if (fabs(vector[i]) < 1e-5) {
                std::cout << std::fixed << std::setprecision(4) << (double) 0 << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(4) << vector[i] << std::endl;
            }
        }
    }

    double &operator[](int i) {
        return vector[i];
    }

    ColumnVector operator+(ColumnVector vector1) {
        ColumnVector result(size());
        for (int i = 0; i < size(); ++i) {
            result[i] = vector[i] + vector1[i];
        }
        return result;
    }

    ColumnVector operator-(ColumnVector vector1) {
        ColumnVector result(size());
        for (int i = 0; i < size(); ++i) {
            result[i] = vector[i] - vector1[i];
        }
        return result;
    }

    ColumnVector operator*(double scalar) {
        ColumnVector result(size());
        for (int i = 0; i < size(); ++i) {
            result[i] = vector[i] * scalar;
        }
        return result;
    }

    double norm() {
        double sum = 0;
        for (int i = 0; i < size(); ++i) {
            sum += vector[i] * vector[i];
        }
        return sqrt(sum);
    }
};

class Matrix {
protected:
    int columns, rows;
    std::vector<ColumnVector> matrix;

public:
    Matrix(int rows, int cols) : columns(cols), rows(rows) {
        matrix.resize(rows, ColumnVector(cols));
    }

    Matrix(int size) : columns(size), rows(size) {
        matrix.resize(size, ColumnVector(size));
    }

    const std::vector<ColumnVector> &getMatrix() const {
        return matrix;
    }

    int number_rows() const {
        return rows;
    }

    int number_cols() const {
        return columns;
    }

    ColumnVector &operator[](int i) {
        return matrix[i];
    };

    ColumnVector operator*(ColumnVector vector) {
        if (columns != vector.size()) {
            std::cout << "Error: the dimensional problem occurred" << std::endl;
            ColumnVector result(0);
            return result;
        }
        ColumnVector result(rows);
        for (int i = 0; i < rows; ++i) {
            double sum = 0;
            for (int j = 0; j < columns; ++j) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    Matrix operator*(Matrix otherMatrix) {
        if (columns != otherMatrix.rows) {
            std::cout << "Error: the dimensional problem occurred" << std::endl;
            Matrix result(0, 0);
            return result;
        }
        Matrix result(rows, otherMatrix.columns);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < otherMatrix.columns; ++j) {
                for (int k = 0; k < otherMatrix.rows; k++) {
                    result[i][j] += matrix[i][k] * otherMatrix[k][j];
                }
            }
        }
        return result;
    }

    Matrix operator+(Matrix otherMatrix) {
        if (rows != otherMatrix.rows || columns != otherMatrix.columns) {
            std::cout << "Error: the dimensional problem occurred" << std::endl;
            Matrix result(0, 0);
            return result;
        }
        Matrix result(rows, columns);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                result[i][j] = matrix[i][j] + otherMatrix[i][j];
            }
        }
        return result;
    }

    Matrix operator-(Matrix otherMatrix) {
        if (rows != otherMatrix.rows || columns != otherMatrix.columns) {
            std::cout << "Error: the dimensional problem occurred" << std::endl;
            Matrix result(0, 0);
            return result;
        }
        Matrix result(rows, columns);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                result[i][j] = matrix[i][j] - otherMatrix[i][j];
            }

        }
        return result;
    }

    Matrix transpose() {
        Matrix result(columns, rows);
        for (int i = 0; i < columns; ++i) {
            for (int j = 0; j < rows; ++j) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }

    void print() {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void printDouble() {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                if (fabs(matrix[i][j]) < 1e-5) {
                    std::cout << std::fixed << std::setprecision(4) << (double) 0 << " ";
                } else {
                    std::cout << std::fixed << std::setprecision(4) << matrix[i][j] << " ";
                }
            }
            std::cout << " " << std::endl;
        }
    }

    void printAugmentedMatrix(Matrix otherMatrix) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                std::cout << std::fixed << std::setprecision(4) << matrix[i][j] << " ";
            }
            for (int j = 0; j < columns; ++j) {
                std::cout << std::fixed << std::setprecision(4) << otherMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void rref_form(ColumnVector &b) {

        for (int i = 0; i < rows; i++) {
            int pivot = i;
            for (int j = i + 1; j < rows; j++) {
                if (abs(matrix[j][i]) > abs(matrix[pivot][i])) {
                    pivot = j;
                }
            }
            std::swap(matrix[i], matrix[pivot]);
            std::swap(b[i], b[pivot]);

            if (fabs(matrix[i][i]) < 1e-10) {
                continue;
            }

            double pivot_val = matrix[i][i];
            matrix[i] = matrix[i] * (1 / pivot_val);
            b[i] = b[i] * (1 / pivot_val);


            for (int j = i + 1; j < rows; j++) {
                double factor = matrix[j][i] / matrix[i][i];
                matrix[j] = matrix[j] - (matrix[i] * factor);
                b[j] = b[j] - (b[i] * factor);
            }
        }
        for (int i = rows - 1; i >= 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                double factor = matrix[j][i] / matrix[i][i];
                matrix[j] = matrix[j] - (matrix[i] * factor);
                b[j] = b[j] - (b[i] * factor);
            }
        }
    }

    int type_of_solution(ColumnVector &b) {
        // if there is zero row and b value of this row is also zero, there is infinity solution since one variable has no unique solution
        // if there is zero row and b value of this row is not zero, there is no solution, because 0 != a, where a != 0;
        // if there is not zero row, there is unique solution
        // There could be situation when many rows with inf solutions and on the top of them there is row with no solution
        bool inf_sol = false;
        int pivots = 0;
        for (int j = rows - 1; j >= 0; j--) {
            for (int i = 0; i < columns; ++i) {
                if (matrix[j][i] != 0) {
                    pivots++;
                }
            }
            if (pivots == 0 && b[rows - 1] != 0) {
                return -1; // no solutions
            } else if (pivots == 0) {
                inf_sol = true;
            }
        }
        if (inf_sol) {
            return 0; // infinity solutions
        } else {
            return 1; // unique solutions
        }
    }

};

class IdentityMatrix : public Matrix {
public:
    IdentityMatrix(int size) : Matrix(size) {
        for (int i = 0; i < size; ++i) {
            matrix[i][i] = 1;
        }
    }
};

class EliminationMatrix : public Matrix {
public:
    EliminationMatrix(int size, int i, int j, Matrix givenMatrix) : Matrix(size) {
        matrix = IdentityMatrix(size).getMatrix();
        double scalar = givenMatrix[i][j] / givenMatrix[j][j];
        matrix[i] = matrix[i] - (matrix[j] * scalar);
    }
};

class PermutationMatrix : public Matrix {
public:
    PermutationMatrix(int size, int i, int j) : Matrix(size) {
        for (int k = 0; k < size; ++k) {
            if (k == i) {
                matrix[j][k] = 1;
            } else if (k == j) {
                matrix[i][k] = 1;
            } else {
                matrix[k][k] = 1;
            }
        }
    }
};

class GaussianElimination {
private:
    Matrix A, B;
    int n;
    ColumnVector b;
    int index;

public:
    GaussianElimination(Matrix A, Matrix B, ColumnVector b) : A(A), n(A.number_rows()), B(B), b(b) {}

    void permute(int i, int j) {
        if (i == j) {
            return;
        }
        PermutationMatrix P(n, i, j);
        A = P * A;
        B = P * B;
        //b = P * b;
    }

    void eliminate(int i, int j) {
        EliminationMatrix E(n, j, i, A);

        A = E * A;
        B = E * B;
        //b = E * b;
    }

    void normalize() {
        double factor;
        for (int i = 0; i < A.number_rows(); i++) {
            factor = 1 / A[i][i];
            A[i] = A[i] * factor;
            B[i] = B[i] * factor;
            //b[i] = b[i]*factor;
        }
    }

    void determinant() {
        double determinant = 1;
        int iterator = 1;
        for (int i = 0; i < A.number_rows(); i++) {
            int pivot = i;
            for (int j = i + 1; j < A.number_rows(); j++) {
                if (abs(A[j][i]) > abs(A[pivot][i])) {
                    pivot = j;
                }
            }

            if (pivot != i) {
                permute(i, pivot);
                determinant *= -1;
                std::cout << "step #" << iterator << ": permutation" << std::endl;
                iterator++;
                A.printDouble();
            }

            determinant *= A[i][i];

            if (fabs(A[i][i]) < 1e-10) {
                continue;
            }

            for (int j = i + 1; j < A.number_rows(); j++) {
                eliminate(i, j);
                std::cout << "step #" << iterator << ": elimination" << std::endl;
                iterator++;
                A.printDouble();
            }

        }
        std::cout << "result:" << std::endl << determinant << std::endl;
    }

    Matrix inverse() {
        index = 0;
        //std::cout << "step #" << index << ": Augmented Matrix" << std::endl;
        index++;
        //A.printAugmentedMatrix(B);
        //std::cout << "Direct way:" << std::endl;
        for (int i = 0; i < A.number_rows(); i++) {
            int pivot = i;
            for (int j = i + 1; j < A.number_rows(); j++) {
                if (fabs(A[j][i]) > fabs(A[pivot][i])) {
                    pivot = j;
                }
            }

            if (pivot != i) {
                permute(pivot, i);
                //std::cout << "step #" << index << ": permutation" << std::endl;
                index++;
                //A.printAugmentedMatrix(B);
            }

            if (fabs(A[i][i]) < 1e-10) {
                continue;
            }

            for (int j = i + 1; j < A.number_rows(); j++) {
                if (A[j][i] == 0) {
                    continue;
                }
                eliminate(i, j);
                //std::cout << "step #" << index << ": elimination" << std::endl;
                index++;
                //A.printAugmentedMatrix(B);
            }
        }

        //std::cout << "Way back:" << std::endl;
        for (int i = A.number_rows() - 1; i >= 0; i--) {
            if (fabs(A[i][i]) < 1e-10) {
                continue;
            }

            for (int j = i - 1; j >= 0; j--) {
                if (A[j][i] == 0) {
                    continue;
                }
                eliminate(i, j);
                //std::cout << "step #" << index << ": elimination" << std::endl;
                index++;
                //A.printAugmentedMatrix(B);
            }
        }

        normalize();
        //std::cout << "Diagonal normalization:" << std::endl;
        index++;
        //A.printAugmentedMatrix(B);

        //std::cout << "result:" << std::endl;
        index++;
        B.printDouble();
        return B;
    }

    void linearEquation() {
        index = 0;
        std::cout << "step #" << index << ":" << std::endl;
        index++;
        A.printDouble();
        b.printDouble();
        for (int i = 0; i < A.number_rows(); i++) {
            int pivot = i;
            for (int j = i + 1; j < A.number_rows(); j++) {
                if (fabs(A[j][i]) > fabs(A[pivot][i])) {
                    pivot = j;
                }
            }

            if (pivot != i) {
                permute(pivot, i);
                std::cout << "step #" << index << ": permutation" << std::endl;
                index++;
                A.printDouble();
                b.printDouble();
            }

            if (fabs(A[i][i]) < 1e-10) {
                continue;
            }

            for (int j = i + 1; j < A.number_rows(); j++) {
                if (A[j][i] == 0) {
                    continue;
                }
                eliminate(i, j);
                std::cout << "step #" << index << ": elimination" << std::endl;
                index++;
                A.printDouble();
                b.printDouble();
            }
        }

        for (int i = A.number_rows() - 1; i >= 0; i--) {
            if (fabs(A[i][i]) < 1e-10) {
                continue;
            }

            for (int j = i - 1; j >= 0; j--) {
                if (A[j][i] == 0) {
                    continue;
                }
                eliminate(i, j);
                std::cout << "step #" << index << ": elimination" << std::endl;
                index++;
                A.printDouble();
                b.printDouble();
            }
        }

        normalize();
        std::cout << "Diagonal normalization:" << std::endl;
        index++;
        A.printDouble();
        b.printDouble();

        std::cout << "result:" << std::endl;

        int solution = type_of_solution();
        if (solution == 1) {
            b.printDouble();
        } else if (solution == 0) {
            std::cout << "INF" << std::endl;
        } else {
            std::cout << "NO" << std::endl;
        }
    }

    void linearEquationWithoutSteps() {
        for (int i = 0; i < A.number_rows(); i++) {
            int pivot = i;
            for (int j = i + 1; j < A.number_rows(); j++) {
                if (fabs(A[j][i]) > fabs(A[pivot][i])) {
                    pivot = j;
                }
            }

            if (pivot != i) {
                permute(pivot, i);
            }

            if (fabs(A[i][i]) < 1e-10) {
                return;
            }

            for (int j = i + 1; j < A.number_rows(); j++) {
                if (A[j][i] == 0) {
                    continue;
                }
                eliminate(i, j);
            }
        }

        for (int i = A.number_rows() - 1; i >= 0; i--) {
            if (fabs(A[i][i]) < 1e-10) {
                return;
            }

            for (int j = i - 1; j >= 0; j--) {
                if (A[j][i] == 0) {
                    continue;
                }
                eliminate(i, j);
            }
        }

        normalize();

        int solution = type_of_solution();

        if (solution == 1) {
            for (int i = 0; i < A.number_rows(); ++i) {
                if (fabs(b[i]) < 1e-10) {
                    std::cout << std::fixed << std::setprecision(2) << (double) 0 << std::endl;
                } else {
                    std::cout << std::fixed << std::setprecision(2) << b[i] << std::endl;
                }
            }
        } else if (solution == 0) {
            std::cout << "INF" << std::endl;
        } else {
            std::cout << "NO" << std::endl;
        }


    }

    int type_of_solution() {
        // if there is zero row and b value of this row is also zero, there is infinity solution since one variable has no unique solution
        // if there is zero row and b value of this row is not zero, there is no solution, because 0 != a, where a != 0;
        // if there is not zero row, there is unique solution
        // There could be situation when many rows with inf solutions and on the top of them there is row with no solution
        bool inf_sol = false;
        int pivots = 0;
        for (int j = A.number_rows() - 1; j >= 0; j--) {
            for (int i = 0; i < A.number_cols(); ++i) {
                if (A[j][i] != 0) {
                    pivots++;
                }
            }
            if (pivots == 0 && b[A.number_rows() - 1] != 0) {
                return -1; // no solutions
            } else if (pivots == 0) {
                inf_sol = true;
            }
        }
        if (inf_sol) {
            return 0; // infinity solutions
        } else {
            return 1; // unique solutions
        }
    }

    ColumnVector leastSquares() {
        std::cout << "A:" << std::endl;
        A.printDouble();

        std::cout << "A_T*A:" << std::endl;
        Matrix AtA = (A.transpose() * A);
        AtA.printDouble();

        std::cout << "(A_T*A)^-1:" << std::endl;
        GaussianElimination gaussianElimination = GaussianElimination(AtA, B, b);
        Matrix AtAInv = gaussianElimination.inverse();

        std::cout << "A_T*b:" << std::endl;
        ColumnVector Atb = (A.transpose() * b);
        Atb.printDouble();

        std::cout << "x~:" << std::endl;
        (AtAInv * Atb).printDouble();
        return (AtAInv * Atb);
    }
};

double min(std::vector<double> values) {
    double result = INT16_MAX;
    for (int i = 0; i < values.size()-1; ++i) {
        if (result > values[i]) result = values[i];
    }
    return result;
}

double max(std::vector<double> values) {
    double result = INT16_MIN;
    for (int i = 0; i < values.size()-1; ++i) {
        if (result < values[i]) result = values[i];
    }
    return result;
}

void solveDifferentialEquation(double alpha1, double alpha2, double beta1, double beta2, std::vector<double>& timeMoments, std::vector<double>& victims, std::vector<double>& killers, double numberPoints, double timeLimit, std::string& function1, std::string& function2) {
    Matrix A = Matrix(2);
    A[0][0] = alpha1;
    A[0][1] = -alpha2;
    A[1][0] = -beta1;
    A[1][1] = beta2;
    ColumnVector x = ColumnVector(2);
    double v_ = alpha2/beta2; // v value of one of the stationary point
    double k_ = alpha1/beta1; // k value of one of the stationary point
    double dfdv = 0;
    double dfdk = - alpha2 * beta1 / beta2;
    double dgdv = beta2 * alpha1 / beta1;
    double dgdk = 0;
    Matrix Jacobian = Matrix(2);
    Jacobian[0][0] = dfdv; // 0
    Jacobian[0][1] = dfdk; // -beta1*alpha2/beta2
    Jacobian[1][0] = dgdv; // 0
    Jacobian[1][1] = dgdk; // -beta2*alpha1/beta1
    double eugeneVal1 = - sqrt(alpha1*alpha2); // -i*sqrt(alpha1*alpha2)
    double eugeneVal2 = sqrt(alpha1*alpha2);  // i*sqrt(alpha1*alpha2)
    ColumnVector q1 = ColumnVector(2); // first eugeneVector
    q1[0] = sqrt(alpha1*alpha2) * beta1/(beta2*alpha1); // i*sqrt(alpha1*alpha2) * beta1/(beta2*alpha1)
    q1[1] = 1; // 1
    ColumnVector q2 = ColumnVector(2); // second eugeneVector
    q2[0] = -q1[0]; // -i*sqrt(alpha1*alpha2) * beta1/(beta2*alpha1)
    q2[1] = q1[1]; // 1
    double v0 = victims[0]-v_;
    double k0 = killers[0]-k_;
    double const1MinusConst2 = -v0*beta2*alpha1 / (sqrt(alpha1*alpha2)*beta1); // v0*beta2*alpha1 / (-i*sqrt(alpha1*alpha2)*beta1);
    double const1PlusConst2 = k0;
    double const1 = (const1PlusConst2 + const1MinusConst2) / 2;
    double const2 = k0 - const1;

    q1[0] = -q1[0]*const1;
    q1[1] = q1[1]*const1;
    q2[0] = -q2[0]*const2;
    q2[1] = q2[1]*const2;

    /*ColumnVector x_final = ColumnVector(2);
    x[0] = 2*q1[0]*cos*/

    for (int i = 1; i < numberPoints+1; i++) {
        timeMoments[i+1] = i*(timeLimit/numberPoints);
        timeMoments.insert(timeMoments.begin()+i, i*(timeLimit/numberPoints));
        double victim = v_ + 2*q1[0]*cos(eugeneVal2*timeMoments[i]);
        double killer = k_ + 2*q2[1]*sin(eugeneVal2*timeMoments[i]);
        victim = v_ + v0*cos(eugeneVal2*timeMoments[i]) - k0*sqrt(alpha2)*beta1* sin(eugeneVal2*timeMoments[i])/beta2/sqrt(alpha1);
        killer = k_ + v0*sqrt(alpha1)*beta2*sin(eugeneVal2*timeMoments[i])/beta1/sqrt(alpha2) + k0*cos(eugeneVal2*timeMoments[i]);
        victims.insert(victims.begin()+i, victim);
        killers.insert(killers.begin()+i, killer);
    }
    function1 = std::to_string(v_) + "+" + std::to_string(v0) + "*cos(" + std::to_string(eugeneVal2) + "*t)" + "-" +
                std::to_string(k0*sqrt(alpha2)*beta1) + "*sin(" + std::to_string(eugeneVal2) + "*t)" + "/" +
                std::to_string(beta2* sqrt(alpha1));
    function2 = std::to_string(k_) + "+" + std::to_string(k0) + "*cos(" + std::to_string(eugeneVal2) + "*t)" + "+" +
                std::to_string(v0*sqrt(alpha1)*beta2) + "*sin(" + std::to_string(eugeneVal2) + "*t)" + "/" +
                std::to_string(beta1* sqrt(alpha2));
}

void scan() {
#ifdef WIN32
    FILE *pipe = _popen(GNUPLOT_NAME, "w");
#else
    FILE* pipe = popen(GNUPLOT_NAME, "w");
#endif
    if (pipe != NULL) {
        double victim0, killer0, timeLimit, numberPoints, alpha1, alpha2, beta1, beta2;

        std::cin >> victim0 >> killer0 >> alpha1 >> beta1 >> alpha2 >> beta2 >> timeLimit >> numberPoints;

        std::vector<double> timeMoments(numberPoints+1), victims(numberPoints+1), killers(numberPoints+1);
        timeMoments[0] = 0;
        victims[0] = victim0;
        killers[0] = killer0;
        double factor = timeLimit/numberPoints;
        std::string function1, function2;

        solveDifferentialEquation(alpha1, alpha2, beta1, beta2, timeMoments, victims, killers, numberPoints, timeLimit, function1, function2);

        std::cout << "t:" << std::endl;
        for (int i = 0; i < numberPoints+1; ++i) {
            if (i == numberPoints) {
                std::cout << std::fixed << std::setprecision(2) << timeMoments[i] << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(2) << timeMoments[i] << " ";
            }
        }

        std::cout << "v:" << std::endl;
        for (int i = 0; i < numberPoints+1; ++i) {
            if (i == numberPoints) {
                std::cout << std::fixed << std::setprecision(2) << victims[i] << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(2) << victims[i] << " ";
            }
        }

        std::cout << "k:" << std::endl;
        for (int i = 0; i < numberPoints+1; ++i) {
            if (i == numberPoints) {
                std::cout << std::fixed << std::setprecision(2) << killers[i] << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(2) << killers[i] << " ";
            }
        }

        fprintf(pipe, "set title 'Lotka-Volterra Equations'\n");
        fprintf(pipe, "set grid\n");
        fprintf(pipe, "set xlabel 'Time'\n");
        fprintf(pipe, "set ylabel 'Population'\n");
        fprintf(pipe, "v(t) = %s\n", function1.c_str());
        fprintf(pipe, "k(t) = %s\n", function2.c_str());
        fprintf(pipe, "set ylabel 'Population'\n");
        double minVict = min(victims), maxVict = max(victims);
        double minKill = min(killers), maxKill = max(killers);
        //fprintf(pipe, "set yrange[%d:%d]\n", min(victims) - 2, max(killers) + 2);
/*        fprintf(pipe, "set xrange[%d:%d]\n", min(X) - 2, max(X) + 2); */
        fprintf(pipe, "set multiplot layout 2,1 rows\n");
        fprintf(pipe, "plot [t=0:%lf] v(t) title 'Prey' with lines, k(t) title 'Predator' with lines\n", timeLimit);
        fprintf(pipe, "set xlabel 'Prey'\n");
        fprintf(pipe, "set ylabel 'Predator'\n");
        fprintf(pipe, "plot '-' title 'Predator' with lines\n", timeLimit);

        for (int i = 0; i < numberPoints; ++i) {
            double predator = killers[i];
            double prey = victims[i];
            fprintf(pipe, "%f\t%f\n", prey, predator);
        }
        fprintf(pipe, "%s\n", "e");
        fflush(pipe);
#ifdef WIN32
        _pclose(pipe);
#else
        pclose(pipe);
#endif
    } else {
        std::cout << "Could not open pipe" << std::endl;
    }
}

int main() {
    scan();
    return 0;
}