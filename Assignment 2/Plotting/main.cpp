#include <iostream>
#include <cstdio>
#include <fstream>
#include <math.h>

/**
    Second Joint Assignment.
    #2.
    @author Danis Alukaev BS-19-02
**/

using namespace std;

#ifdef WIN32
#define GNUPLOT_NAME "C:\\gnuplot\\bin\\gnuplot -persist"
#endif // WIN32

/**
 * Class Matrix.
 * Represents a rectangular array of numbers arranged in rows and columns.
 */
class Matrix
{
public:
    int n, m; // dimensions of matrix
    // int swaps = 0; // number of permutations during the elimination
    double **data; // dynamic array to store elements of matrix

    /**
    * Constructor of the class Matrix.
    * Dynamically allocates memory to store the matrix with the received number of rows and columns.
    *
    * @param rows - number of rows of a matrix.
    * @param columns - number of columns of a matrix.
    */
    Matrix(int rows, int columns)
    {
        n = rows; // set number of rows
        m = columns; // set number of columns
        // allocating memory for array of arrays
        data = (double **) malloc(sizeof(double*) * n);
        for(int i = 0; i < n; i++)
            data[i] = (double*)malloc(sizeof(double) * m);
    }

    /**
    * Overloading " >> " operator for a class Matrix
    */
    friend istream& operator >> (istream& in, const Matrix& matrix)
    {
        for(int i = 0; i < matrix.n; i++)
            for(int j = 0; j < matrix.m; j++)
                in >> matrix.data[i][j];
        return in;
    }

    /**
    * Overloading " << " operator for a class Matrix
    */
    friend ostream& operator << (ostream& out, const Matrix& matrix)
    {
        for(int i = 0; i < matrix.n; i++)
        {
            for(int j = 0; j < matrix.m-1; j++)
                out << matrix.data[i][j] << " ";
            out << round(matrix.data[i][matrix.m-1] * 100) / 100 << "\n";
        }
        return out;
    }

    /**
    * Overloading " = " operator for a class Matrix
    *
    * @param other - the matrix to be moved in this instance.
    * @return *this - this instance of a class Matrix.
    */
    Matrix& operator = (Matrix& other)
    {
        n = other.n; // set new
        m = other.m; // dimensions
        // swaps = other.swaps; // transmit number of permutations
        data = other.data; // transfer elements
        return *this;
    }

    /**
    * Overloading " + " operator for a class Matrix
    *
    * @param other - the matrix to be added with this instance.
    * @return *matrixN - sum of two matrices.
    */
    Matrix& operator + (Matrix& other)
    {
        Matrix* matrixN = new Matrix(n, m); // creating instance of class Matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                matrixN -> data[i][j] = data[i][j] + other.data[i][j]; // store result of addition
        return *matrixN;
    }

    /**
    * Overloading " - " operator for a class Matrix
    *
    * @param other - the matrix to be subtracted with this instance.
    * @return *matrixN - difference of two matrices.
    */
    Matrix& operator - (Matrix& other)
    {
        Matrix* matrixN = new Matrix(n, m); // creating instance of class Matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                matrixN -> data[i][j] = data[i][j] - other.data[i][j]; // store result of subtraction
        return *matrixN;
    }

    /**
    * Overloading " * " operator for a class Matrix
    *
    * @param other - the matrix to be multiplied with this instance.
    * @return *matrixN - transposed matrix.
    */
    Matrix& operator * (Matrix& other)
    {
        Matrix* product = new Matrix(n, other.m); // creating instance of class Matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j < other.m; j++)
                product -> data[i][j] = 0; // nullify all positions of new matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j< other.m; j++)
                for(int k = 0; k < m; k++)
                    product -> data[i][j] += data[i][k] * other.data[k][j]; // store the obtained in multiplication values
        return *product;
    }

    /**
    * Transposition of the matrix.
    * Flips a matrix over its diagonal.
    *
    * @return *matrixN - transposed matrix.
    */
    Matrix& transpose()
    {
        Matrix* matrixN = new Matrix(m, n); // creating instance of class Matrix
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                matrixN -> data[i][j] = data[j][i]; // store elements of rows in corresponding columns
        return *matrixN;
    }

    /**
    * Destructor of the class Matrix.
    */
    ~Matrix()
    {
        for(int i = 0; i < n; i++)
            delete [] data[i];
        delete [] data;
    }
};

/**
 * Class SquareMatrix.
 * Indeed, represents the matrix with the same number of rows and columns.
 */
class SquareMatrix : public Matrix
{
public:
    /**
    * Constructor of the class SquareMatrix.
    * Creates the matrix with the same number of rows and columns.
    *
    * @param dimension - dimension of matrix.
    */
    SquareMatrix (int dimension):Matrix(dimension, dimension)
    {
        // creating new instance of class Matrix with received number of rows and columns
    }
};

/**
 * Class IdentityMatrix.
 * Indeed, represents the square matrix with ones on the main diagonal and zeros elsewhere.
 */
class IdentityMatrix : public SquareMatrix
{
public:
    /**
    * Constructor of the class IdentityMatrix.
    * Creates the square matrix with ones on the main diagonal and zeros elsewhere.
    *
    * @param dimension - dimension of identity matrix.
    */
    IdentityMatrix (int dimension):SquareMatrix(dimension)
    {
        for(int i = 0; i < dimension; i++)
            for(int j = 0; j < dimension; j++)
                i == j ? data[i][j] = 1 : data [i][j] = 0; // creating identity matrix, fill 0 in all positions except main diagonal
    }
};

/**
 * Class PermutationMatrix.
 * Particularly, represents the square matrix used to exchange two rows of the received matrix.
 */
class PermutationMatrix : public SquareMatrix
{
public:
    /**
    * Constructor of the class PermutationMatrix.
    * Creates the identity matrix with exchanged columns i1 and i2.
    *
    * @param dimension - dimension of permutation matrix.
    * @param i1 - the first column to be exchanged.
    * @param i2 - the second column to be exchanged
    */
    PermutationMatrix (int dimension, int i1 = 1, int i2 = 1):SquareMatrix(dimension)
    {
        i1--; // number of lines belongs
        i2--; // to [1; +inf]
        for(int i = 0; i < dimension; i++)
            for(int j = 0; j < dimension; j++)
                i == j ? data[i][j] = 1 : data [i][j] = 0; // creating identity matrix, fill 0 in all positions except main diagonal
        data[i2][i2] = 0; // swap corresponding
        data[i2][i1] = 1; // elements of lines
        data[i1][i1] = 0; // to convert it into
        data[i1][i2] = 1; // permutation matrix
    }
};

/**
 * Class EliminationMatrix.
 * Apparently, represents the square matrix used to lead corresponding elements of the received matrix to 0.
 */
class EliminationMatrix : public IdentityMatrix
{
public:
    /**
    * Constructor of the class EliminationMatrix.
    * Creates the matrix that nullify corresponding element of the received matrix.
    *
    * @param matrix - given matrix, which element [i1, i2] should be 0.
    * @param i1 - the element's line of the given matrix.
    * @param i2 - element's the column of the given matrix.
    */
    EliminationMatrix (Matrix& matrix, int i1, int i2):IdentityMatrix(matrix.n)
    {
        i1--; // number of lines belongs
        i2--; // to [1; +inf]
        // check the potential division by 0
        try
        {
            if (matrix.data[i2][i2] == 0)
                throw runtime_error("Division by 0");
            data[i1][i2] = - matrix.data[i1][i2] / matrix.data[i2][i2]; // calculate coefficient that will nullify corresponding element
        }
        catch(runtime_error& e)
        {
            cout << e.what() << endl;
        }
    }
};

/**
 * Class ScaleMatrix.
 * Apparently, represents the matrix used to lead diagonal matrix to the identity matrix.
 */
class ScaleMatrix : public Matrix
{
public:
    /**
    * Constructor of the class ScaleMatrix.
    * Creates the matrix which principal diagonal elements are reciprocal to corresponding elements of the received matrix.
    *
    * @param matrix - given matrix, which principal diagonal elements should be 1's.
    */
    ScaleMatrix (Matrix& matrix):Matrix(matrix.n, matrix.n)
    {
        for(int i = 0; i < matrix.n; i++)       // treat all
            for(int j = 0; j < matrix.n; j++)   // elements of created matrix
                data[i][j] = 0; // nullify all elements of matrix
        for(int i = 0; i < matrix.n; i++)
            data[i][i] = 1 / matrix.data[i][i]; // set elements of main diagonal to corresponding coefficients
    }
};

/**
 * Class AugmentedMatrix.
 * In fact, represents matrix that can be used to perform the same elementary row operations on each of the given matrices.
 * Particularly, in this implementation it applied to find inverse matrix.
 */
class AugmentedMatrix : public Matrix
{
public:
    /**
    * Constructor of the class AugmentedMatrix.
    * Merges received and identity matrices by appending their columns.
    *
    * @param matrix - given matrix to be to be merged with identity matrix.
    */
    AugmentedMatrix(Matrix& matrix) : Matrix(matrix.n, 2 * matrix.n)
    {
        for(int i = 0; i < matrix.n; i++)
        {
            for(int j = 0; j < matrix.n; j++) // treat all columns from 0 up to n-th
                data[i][j] = matrix.data[i][j]; // copy elements of received matrix
            for(int j = matrix.n; j < 2 * matrix.n; j++) // treat all columns from n-th up to 2*n-th
                i == (j - matrix.n) ? data[i][j] = 1 : data [i][j] = 0; // creating identity matrix, fill 0 in all positions except diagonal
        }
    }
};

/**
 * Inverses the received matrix using Gaussian Elimination approach.
 *
 * @param matrix - given matrix to be inversed.
 * @return inversed - the inversed matrix.
 */
Matrix& getInverse(Matrix& matrix)
{
    Matrix *Augmented = new AugmentedMatrix(matrix); // creating an augmented matrix
    int step = 1, swaps = 0; // initialization number of steps and permutations
    // nullify elements under the diagonal
    for(int i = 0; i < Augmented->n; i++)  // treat all rows of matrix
    {
        // find pivot with maximum absolute value
        // store its index in pivotIndex
        // store its value in pivotValue
        int pivotIndex = i;
        double pivotValue = abs(Augmented->data[i][i]);
        for(int j = i; j < Augmented->n; j++)
        {
            if (pivotValue < abs(Augmented->data[j][i]) && ((abs(Augmented->data[j][i]) - pivotValue) >= 0.01)) // find pivot with maximum absolute value
            {
                pivotIndex = j; // store index of the found matrix
                pivotValue = abs(Augmented->data[j][i]); // store value of the found matrix
            }
        }
        // swap the current line with the found pivot line
        if(pivotIndex != i)
        {
            Matrix *P = new PermutationMatrix(Augmented->n, pivotIndex + 1, i + 1); // create the permutation matrix P_{pivotline+1 i+1} for a current state
            *Augmented = *P * (*Augmented); // apply the permutation matrix
            swaps++; // increment the number of permutations
        }
        for(int j = i + 1; j < Augmented->n; j++)
        {
            Matrix *E = new EliminationMatrix(*Augmented, j + 1, i + 1); // create the elimination matrix E_{j+1 i+1} for a current state
            *Augmented = *E * (*Augmented); // apply the elimination matrix
        }
    }
    // nullify elements over the diagonal
    for(int i = Augmented->n-1; i >= 0; i--)
    {
        for(int j = i - 1; j >= 0; j--)
        {
            Matrix *E = new EliminationMatrix(*Augmented, j + 1, i + 1); // create the elimination matrix E_{j+1 i+1} for a current state
            *Augmented = *E * (*Augmented); // apply the elimination matrix
        }
    }
    // Diagonal normalization
    Matrix *scale = new ScaleMatrix(*Augmented); // create the scale matrix for the diagonal normalization
    *Augmented = *scale * (*Augmented); // diagonal normalization
    Matrix *inversed = new SquareMatrix(Augmented->n);
    // move the right part from n-th up to 2*n-th of the augmented matrix to a created matrix "inversed"
    for(int i = 0; i < Augmented->n; i++)
        for(int j = Augmented->n; j < 2*Augmented->n; j++)
            inversed -> data[i][j - Augmented->n] = Augmented->data[i][j];
    return *inversed; // return the inversed matrix
}

/**
 * Computes approximate solution for a given system of linear equations.
 * It can be performed by applying Ordinary least squares (OLS) estimator:
 * x' = ( A_transposed * A )^{-1} * A_transposed * b, where x' is the estimated value of the unknown parameter vector.
 *
 * @param A - vector consisting of regressors (n-dimensional column-vectors).
 * @param b - vector consisting of regressands.
 * @return optimalSolution - the estimated value of the unknown parameter vector.
 */
Matrix& approximateSolution(Matrix& A, Matrix& b)
{
    Matrix A_transposed_A = A.transpose() * A; // calculate A_transposed * A
    cout << "A_T*A:\n" << A_transposed_A;
    Matrix InverseMatrix = getInverse(A_transposed_A); // get inverse matrix for A_transposed * A
    cout << "(A_T*A)^-1:\n" << InverseMatrix;
    Matrix A_transposed_b = A.transpose() * b; // calculate A_transposed * b
    cout << "A_T*b:\n" << A_transposed_b;
    return InverseMatrix * A_transposed_b; // compute and return an optimal solution for a given system of equations
}

int main()
{
#ifdef WIN32
    FILE* pipe = _popen(GNUPLOT_NAME, "w");
#endif
    ifstream inFile;
    inFile.open("data.dat"); // try to access file data.dat
    if(!inFile.fail()) // if file founded
    {
        cout.setf(ios::fixed); // set format of
        cout.precision(2);     // output values
        const int M = 43, N = 2; // set number of points and polynomial degree of approximation
        Matrix *input = new Matrix(M, 2); // create new instance of class Matrix to store experimental data
        inFile >> *input; // read input
        Matrix *A = new Matrix(M, N+1); // create new instance of class Matrix to store regressors
        for(int i = 0; i < M; i++)
            for(int j = 0; j < N+1; j++)
                A->data[i][j] = pow(input->data[i][0], j); // place n-th power of corresponding t_{i} in matrix A
        cout << "A:\n" << *A;
        Matrix *b = new Matrix(M, 1); // create new instance of class Matrix to store regressands
        for(int i = 0; i < M; i++)
            b->data[i][0] = input->data[i][1]; // move regressands to created container
        Matrix coefficients = approximateSolution(*A, *b);
        cout << "x~:\n" << coefficients; // print the optimal solution of the system of linear equations
        if(pipe != NULL) // if pipeline established
        {
            fprintf(pipe, "%s\n", "set grid\nset xlabel 'Values of x'\nset ylabel 'Values of y'\n"); // set grid and labels for axes
            fprintf(pipe, "%s\n", "cd 'C:\\Users\\pc\\Desktop\\JA_2 Alukaev'"); // access the directory with the dataset
            // plot the experimental data and parabolic approximation for it
            fprintf(pipe, "%s%f%s%f%s%f%s\n", "plot 'data.dat' using 1:2 title 'Experimental data' with points pointtype 5, ",
                    coefficients.data[2][0], "*x**2+(", coefficients.data[1][0], ")*x+(", coefficients.data[0][0], ") title 'Approximation'");
            fflush(pipe);
        }
#ifdef WIN32
        _pclose(pipe);
#endif // WIN32
    }
    return 0;
}
