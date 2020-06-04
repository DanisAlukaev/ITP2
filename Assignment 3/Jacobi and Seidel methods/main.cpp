#include <iostream>
#include <cmath>

#define INF (unsigned)!((int)0)

using namespace std;

/**
    Third Joint Assignment.
    #1,2.
    @author Danis Alukaev BS-19-02.
**/

/**
 * Class Matrix.
 * Represents a rectangular array of numbers arranged in rows and columns.
 */
class Matrix
{
public:
    int n, m; // dimensions of a matrix
    double **data; // the dynamic array to store elements of a matrix

    /**
    * Constructor of the class Matrix.
    * Dynamically allocates memory to store the matrix with the received number of rows and columns.
    *
    * @param rows - the number of rows of a matrix.
    * @param columns - the number of columns of a matrix.
    */
    Matrix(int rows, int columns)
    {
        n = rows; // set the number of rows
        m = columns; // set the number of columns
        // allocate memory for an array of arrays
        data = (double **) malloc(sizeof(double*) * n);
        for(int i = 0; i < n; i++)
            data[i] = (double*)malloc(sizeof(double) * m);
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                data[i][j] = 0;
    }

    /**
    * Overloading " >> " operator for a class Matrix
    */
    friend istream& operator >> (istream& in, const Matrix& matrix)
    {
        for(int i = 0; i < matrix.n; i++)
            for(int j = 0; j < matrix.m; j++)
                in >> matrix.data[i][j]; // read the element with indexes i, j
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
            out << matrix.data[i][matrix.m-1] << "\n"; // print the element with indexes i, j
            // use the construction round(someNumber * 100) / 100 to round half towards one
        }
        return out;
    }

    /**
    * Overloading " = " operator for a class Matrix
    *
    * @param other - the matrix to be moved to this instance.
    * @return *this - this instance of a class Matrix.
    */
    Matrix& operator = (Matrix& other)
    {
        n = other.n; // set new dimensions
        m = other.m; // of a matrix
        data = other.data; // transfer elements to this instance of a matrix
        return *this;
    }

    /**
    * Overloading " + " operator for a class Matrix
    *
    * @param other - the matrix to be added to this instance.
    * @return *matrixN - the sum of two matrices.
    */
    Matrix& operator + (Matrix& other)
    {
        Matrix* matrixN = new Matrix(n, m); // creating new instance of the class Matrix to store the result
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                matrixN -> data[i][j] = data[i][j] + other.data[i][j]; // store the result of an addition
        return *matrixN;
    }

    /**
    * Overloading " - " operator for a class Matrix
    *
    * @param other - the matrix to be subtracted from this instance.
    * @return *matrixN - the difference of two matrices.
    */
    Matrix& operator - (Matrix& other)
    {
        Matrix* matrixN = new Matrix(n, m); // creating new instance of the class Matrix to store the result
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                matrixN -> data[i][j] = data[i][j] - other.data[i][j]; // store the result of a subtraction
        return *matrixN;
    }

    /**
    * Overloading " * " operator for a class Matrix
    *
    * @param other - the matrix to be multiplied by this instance.
    * @return *matrixN - the transposed matrix.
    */
    Matrix& operator * (Matrix& other)
    {
        Matrix* product = new Matrix(n, other.m); // creating new instance of the class Matrix to store the result
        for(int i = 0; i < n; i++)
            for(int j = 0; j < other.m; j++)
                product -> data[i][j] = 0; // nullify all positions of a new matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j< other.m; j++)
                for(int k = 0; k < m; k++)
                    product -> data[i][j] += data[i][k] * other.data[k][j]; // store the result of multiplication
        return *product;
    }

    /**
    * Transposition of the matrix.
    * Flips a matrix over its principal diagonal.
    *
    * @return *matrixN - the transposed matrix.
    */
    Matrix& transpose()
    {
        Matrix* matrixN = new Matrix(m, n); // creating new instance of the class Matrix to store the result
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                matrixN -> data[i][j] = data[j][i]; // store elements of a particular row in the corresponding column
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
 * Represents the matrix with the same number of rows and columns.
 */
class SquareMatrix : public Matrix
{
public:
    /**
    * Constructor of the class SquareMatrix.
    * Creates the matrix with the same number of rows and columns.
    *
    * @param dimension - the dimension of matrix.
    */
    SquareMatrix (int dimension) : Matrix(dimension, dimension)
    {
        // creating new instance of the class Matrix with the received number of rows and columns
    }
};

/**
 * Class IdentityMatrix.
 * Represents the square matrix with ones on the main diagonal and zeros elsewhere.
 */
class IdentityMatrix : public SquareMatrix
{
public:
    /**
    * Constructor of the class IdentityMatrix.
    * Creates the square matrix with ones on the main diagonal and zeros elsewhere.
    *
    * @param dimension - the dimension of an identity matrix.
    */
    IdentityMatrix (int dimension) : SquareMatrix(dimension)
    {
        for(int i = 0; i < dimension; i++)
            for(int j = 0; j < dimension; j++)
                i == j ? data[i][j] = 1 : data [i][j] = 0; // creating the identity matrix, set the main diagonal elements to ones and fill the rest of matrix with zeroes
    }
};

/**
 * Class PermutationMatrix.
 * Represents the square matrix used to exchange two rows with received indexes of the matrix.
 */
class PermutationMatrix : public SquareMatrix
{
public:
    /**
    * Constructor of the class PermutationMatrix.
    * Creates the identity matrix with exchanged columns i1 and i2.
    *
    * @param dimension - the dimension of a permutation matrix.
    * @param i1 - the first column to be exchanged.
    * @param i2 - the second column to be exchanged
    */
    PermutationMatrix (int dimension, int i1 = 1, int i2 = 1) : SquareMatrix(dimension)
    {
        i1--; // since the number of lines of matrix in linear algebra belongs
        i2--; // to the range [1; +inf], map it to the [0; +inf]
        for(int i = 0; i < dimension; i++)
            for(int j = 0; j < dimension; j++)
                i == j ? data[i][j] = 1 : data [i][j] = 0; // creating the identity matrix, set the main diagonal elements to ones and fill the rest of matrix with zeroes
        data[i2][i2] = 0; // swap corresponding
        data[i2][i1] = 1; // elements of lines
        data[i1][i1] = 0; // to make it
        data[i1][i2] = 1; // permutation matrix
    }
};

/**
 * Class EliminationMatrix.
 * Represents the square matrix used to lead elements with received indexes of the matrix to zeroes.
 */
class EliminationMatrix : public IdentityMatrix
{
public:
    /**
    * Constructor of the class EliminationMatrix.
    * Creates the matrix that nullify the corresponding element of the received matrix.
    *
    * @param matrix - given matrix, which element [i1, i2] should be zero.
    * @param i1 - the element's line of the given matrix.
    * @param i2 - the element's column of the given matrix.
    */
    EliminationMatrix (Matrix& matrix, int i1, int i2) : IdentityMatrix(matrix.n)
    {
        i1--; // since the number of lines of matrix in linear algebra belongs
        i2--; // to the range [1; +inf], map it to the [0; +inf]
        // check the potential division by 0
        try
        {
            if (matrix.data[i2][i2] == 0)
                throw runtime_error("Division by 0");
            data[i1][i2] = - matrix.data[i1][i2] / matrix.data[i2][i2]; // calculate the coefficient that will nullify the element with received indexes
        }
        catch(runtime_error& e)
        {
            cout << e.what() << endl;
        }
    }
};

/**
 * Class ScaleMatrix.
 * Represents the matrix used to lead the diagonal matrix to the identity matrix.
 */
class ScaleMatrix : public Matrix
{
public:
    /**
    * Constructor of the class ScaleMatrix.
    * Creates the matrix which principal diagonal elements are reciprocal to corresponding elements of the received matrix.
    *
    * @param matrix - the given matrix, which principal diagonal elements should be ones.
    */
    ScaleMatrix (Matrix& matrix) : Matrix(matrix.n, matrix.n)
    {
        for(int i = 0; i < matrix.n; i++)       // treat all
            for(int j = 0; j < matrix.n; j++)   // elements of the created matrix
                data[i][j] = 0; // nullify all elements of a matrix
        for(int i = 0; i < matrix.n; i++)
            data[i][i] = 1 / matrix.data[i][i]; // set elements of the main diagonal to corresponding coefficients
    }
};

/**
 * Class AugmentedMatrix.
 * Represents matrix that can be used to perform the same elementary row operations on each of the given matrices.
 * Particularly, in this implementation it applied to find the inverse matrix.
 */
class AugmentedMatrix : public Matrix
{
public:
    /**
    * Constructor of the class AugmentedMatrix.
    * Merges the received and identity matrices by appending their columns.
    *
    * @param matrix - given matrix to be merged with identity matrix.
    */
    AugmentedMatrix(Matrix& matrix) : Matrix(matrix.n, 2 * matrix.n)
    {
        for(int i = 0; i < matrix.n; i++)
        {
            for(int j = 0; j < matrix.n; j++) // treat all columns from 0 up to n-th
                data[i][j] = matrix.data[i][j]; // copy elements of received matrix
            for(int j = matrix.n; j < 2 * matrix.n; j++) // treat all columns from n-th up to 2*n-th
                i == (j - matrix.n) ? data[i][j] = 1 : data [i][j] = 0; // set the main diagonal elements to ones and fill the rest of matrix with zeroes
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
    int step = 1, swaps = 0; // the number of steps and permutations
    // nullify elements under the principal diagonal
    for(int i = 0; i < Augmented->n; i++)  // treat all rows of a matrix
    {
        // find the pivot with the maximum absolute value
        // store its index in the pivotIndex
        // store its value in the pivotValue
        int pivotIndex = i;
        double pivotValue = abs(Augmented->data[i][i]);
        for(int j = i; j < Augmented->n; j++)
        {
            if (pivotValue < abs(Augmented->data[j][i]) && ((abs(Augmented->data[j][i]) - pivotValue) >= 0.01)) // find the pivot with maximum absolute value
            {
                pivotIndex = j; // store the index of the found element
                pivotValue = abs(Augmented->data[j][i]); // store value of the found element
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
    // nullify elements over the principal diagonal
    for(int i = Augmented->n-1; i >= 0; i--)
    {
        for(int j = i - 1; j >= 0; j--)
        {
            Matrix *E = new EliminationMatrix(*Augmented, j + 1, i + 1); // create the elimination matrix E_{j+1 i+1} for a current state
            *Augmented = *E * (*Augmented); // apply the elimination matrix
        }
    }
    // the diagonal normalization
    Matrix *scale = new ScaleMatrix(*Augmented); // create the scale matrix for the diagonal normalization
    *Augmented = *scale * (*Augmented); // perform the diagonal normalization
    Matrix *inversed = new SquareMatrix(Augmented->n);
    // move the right part from n-th up to 2*n-th column of the augmented matrix to a created "inversed" matrix
    for(int i = 0; i < Augmented->n; i++)
        for(int j = Augmented->n; j < 2*Augmented->n; j++)
            inversed -> data[i][j - Augmented->n] = Augmented->data[i][j];
    return *inversed; // return the inversed matrix
}

/**
 * Auxiliary method to check the convergence of iterative method.
 *
 * @param a - current solution.
 * @param b - previous solution.
 * @return length of the difference of two solutions.
 */
double calculateAccuracy(Matrix& a, Matrix& b)
{
    double sumOfComponents = 0;
    for(int i = 0; i < a.n; i++)
        // calculate the sum of squares of vectors difference components
        sumOfComponents += (a.data[i][0]-b.data[i][0])*(a.data[i][0]-b.data[i][0]);
    return sqrt(sumOfComponents); // square root of the sum of squares of vectors difference components
}

/**
 * Solves the system of linear equations via Jacobi iterative method.
 *
 * @param A - given coefficient matrix.
 * @param b - given column vector of free variables.
 * @param epsilon - required accuracy of a solution.
 * @return the final solution with a given precision.
 */
Matrix& Jacobi(Matrix& A, Matrix& b, double epsilon)
{
    // Check the diagonal predominance, i.e. for every row of the matrix A, the magnitude of the diagonal entry in a row
    // should be greater than or equal to the sum of the magnitudes of all the other (non-diagonal) entries in that row
    for(int i = 0; i < A.n; i++)
    {
        double sum = 0;
        for(int j = 0; j < A.n; j++)
            if(i != j)
                sum += A.data[i][j];
        if(sum > A.data[i][i])
        {
            // if the condition is not fulfilled, then method is not applicable
            cout << "The method is not applicable!";
            exit(0);
        }
    }
    int iteration = 0; // iteration counter
    double accuracy = INF; // accuracy parameter
    Matrix *alpha = new SquareMatrix(A.n); // coefficient matrix alpha
    Matrix *beta = new Matrix(A.n, 1);  // coefficient vector beta
    *beta = b;
    for(int i = 0; i < A.n; i++)
    {
        for(int j = 0; j < A.n; j++)
            // set the diagonal elements of the matrix alpha to zeros and the rest of it to coefficients of the matrix A
            // divided by its corresponding diagonal elements
            i == j ? alpha->data[i][i] = 0 : alpha->data[i][j] = -A.data[i][j]/A.data[i][i];
        // divide elements of the vector b on corresponding diagonal elements of the matrix A
        beta->data[i][0] /= A.data[i][i];
    }
    cout << "alpha:\n" << *alpha;
    cout << "beta:\n" << *beta;

    Matrix *xPrev = new Matrix(A.n, 1); // result of a previous iteration
    Matrix *xCur = new Matrix(A.n, 1);  // result of a current iteration
    *xPrev = *beta;
    while (accuracy > epsilon)
    {
        // calculate X_{i} = beta + alpha * X_{i-1}
        *xCur = *beta + *alpha * *xPrev;
        cout << "x(" << iteration << "):\n";
        cout << *xPrev; // X_{i-1}
        accuracy = calculateAccuracy(*xCur, *xPrev); // find an accuracy of the obtained solution
        cout << "e: " << accuracy << "\n";
        *xPrev = *xCur; // X_{i-1} <- X_{i}
        iteration ++;
    }
    cout << "x(" << iteration << "):\n";
    cout << *xCur; // final solution with a given precision
    return *xCur;
}

/**
 * Solves the system of linear equations via Seidel iterative method.
 *
 * @param A - given coefficient matrix.
 * @param b - given column vector of free variables.
 * @param epsilon - required accuracy of a solution.
 * @return the final solution with a given precision.
 */
Matrix& Seidel(Matrix& A, Matrix& b, double epsilon)
{
    // Check the diagonal predominance, i.e. for every row of the matrix A, the magnitude of the diagonal entry in a row
    // should be greater than or equal to the sum of the magnitudes of all the other (non-diagonal) entries in that row
    for(int i = 0; i < A.n; i++)
    {
        double sum = 0;
        for(int j = 0; j < A.n; j++)
            if(i != j)
                sum += A.data[i][j];
        if(sum > A.data[i][i])
        {
            // if the condition is not fulfilled, then method is not applicable
            cout << "The method is not applicable!";
            exit(0);
        }
    }
    int iteration = 0; // iteration counter
    double accuracy = INF; // accuracy parameter
    Matrix *alpha = new SquareMatrix(A.n); // coefficient matrix alpha
    Matrix *beta = new Matrix(A.n, 1); // coefficient vector beta
    Matrix *B = new SquareMatrix(A.n); // lower triangular matrix formed from the alpha matrix
    Matrix *C = new SquareMatrix(A.n); // upper triangular matrix formed from the alpha matrix
    Matrix* Identity = new IdentityMatrix(A.n); // identity matrix with the given dimension
    Matrix* diffIB = new SquareMatrix(A.n);  // (I-B) matrix
    Matrix* diffIBInversed = new SquareMatrix(A.n); // (I-B)^-1 matrix
    *beta = b;
    for(int i = 0; i < A.n; i++)
    {
        for(int j = 0; j < A.n; j++)
            // set the diagonal elements of the matrix alpha to zeros and the rest of it to coefficients of the matrix A
            // divided by its corresponding diagonal elements
            i == j ? alpha->data[i][i] = 0 : alpha->data[i][j] = -A.data[i][j]/A.data[i][i];
        // divide elements of the vector b on corresponding diagonal elements of the matrix A
        beta->data[i][0] /= A.data[i][i];
    }
    cout << "beta:\n" << *beta;
    cout << "alpha:\n" << *alpha;
    for(int i = 0; i < A.n; i++)
        for(int j = 0; j < A.n; j++)
            // fill the upper and lower triangular matrices C and B with elements of matrix alpha
            i > j ? B->data[i][j] = alpha->data[i][j] : C->data[i][j] = alpha->data[i][j];
    cout << "B:\n" << *B;
    cout << "C:\n" << *C;
    *diffIB = *Identity - (*B); // calculate (I-B) matrix
    cout << "I-B:\n" << *diffIB;
    *diffIBInversed = getInverse(*diffIB); // calculate (I-B)^-1 matrix
    cout << "(I-B)^-1:\n" << *diffIBInversed;

    Matrix *xPrev = new Matrix(A.n, 1); // result of a previous iteration
    Matrix *xCur = new Matrix(A.n, 1);  // result of a current iteration
    *xPrev = *beta;
    while (accuracy > epsilon)
    {
        // calculate X_{i} = (B-I)^-1 * C * X_{i-1} + (B-I)^-1 * beta
        *xCur = *diffIBInversed * *C * *xPrev + *diffIBInversed * *beta;
        cout << "x(" << iteration << "):\n";
        cout << *xPrev; // X_{i-1}
        accuracy = calculateAccuracy(*xCur, *xPrev); // find an accuracy of the obtained solution
        cout << "e: " << accuracy << "\n";
        *xPrev = *xCur; // X_{i-1} <- X_{i}
        iteration ++;
    }
    cout << "x(" << iteration << "):\n";
    cout << *xCur; // final solution with a given precision
    return *xCur;
}

int main()
{
    cout.setf(ios::fixed); // set the decimal precision of
    cout.precision(4);     // output values
    int n; // variable that stores the dimension of the coefficient matrix A
    double epsilon; // variable that stores the approximation accuracy
    cin >> n;
    Matrix *A = new SquareMatrix(n); // create the new instance of the class Matrix to store the coefficient matrix A
    Matrix *b = new Matrix(n, 1); // create the new instance of the class Matrix to store the free vector b
    cin >> *A;
    cin >> n;
    cin >> *b;
    cin >> epsilon;
    // apply Seidel method to calculate a solution of the system of linear equations
    Seidel(*A, *b, epsilon);
}