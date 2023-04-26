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

int min(std::vector<int> values) {
    int result = INT_MAX;
    for (auto i: values) {
        if (result > i) result = i;
    }
    return result;
}

int max(std::vector<int> values) {
    int result = INT_MIN;
    for (auto i: values) {
        if (result < i) result = i;
    }
    return result;
}

void scan() {
#ifdef WIN32
    FILE *pipe = _popen(GNUPLOT_NAME, "w");
#else
    FILE* pipe = popen(GNUPLOT_NAME, "w");
#endif
    if (pipe != NULL) {
        int lenghOfDataSet, degreeOfPolynomial;
        std::cin >> lenghOfDataSet;

        std::vector<int> X;
        std::vector<int> Y;
        ColumnVector b = ColumnVector(lenghOfDataSet);

        int x, y, it = 0;
        for (int i = 0; i < lenghOfDataSet; i++) {
            std::cin >> x >> y;
            X.insert(X.begin() + it, x);
            Y.insert(Y.begin() + it, y);
            b[i] = y;
        }

        fprintf(pipe, "set xrange[%d:%d]\n", min(X) - 2, max(X) + 2);
        fprintf(pipe, "set yrange[%d:%d]\n", min(Y) - 2, max(Y) + 2);
        fprintf(pipe, "set style data lines\n");

        std::cin >> degreeOfPolynomial;

        Matrix A = Matrix(lenghOfDataSet, degreeOfPolynomial + 1);
        IdentityMatrix I = IdentityMatrix(degreeOfPolynomial + 1);
        ColumnVector result = ColumnVector(degreeOfPolynomial);

        for (int i = 0; i < degreeOfPolynomial + 1; ++i) {
            for (int j = 0; j < lenghOfDataSet; ++j) {
                A[j][i] = pow(X[lenghOfDataSet - j - 1], i);
            }
        }

        GaussianElimination gaussianElimination = GaussianElimination(A, I, b);
        result = gaussianElimination.leastSquares();

        std::string function = "";
        for (int j = 0; j < degreeOfPolynomial+1; ++j) {
            if (j == degreeOfPolynomial) {
                function += std::to_string(result[j]) + "*x**" + std::to_string(j);
            } else {
                function += std::to_string(result[j]) + "*x**" + std::to_string(j) + "+";
            }
        }


        fprintf(pipe, "plot '-' title '' with points, %s\n", function.c_str());

        for (int i = 0; i < lenghOfDataSet; ++i) {
            double x = X[i];
            double y = Y[i];
            fprintf(pipe, "%f\t%f\n", x, y);
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