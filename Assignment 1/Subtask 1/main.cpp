#include <iostream>
#include <cmath>

using namespace std;

/**
    #2.1.1
    Danis Alukaev BS-19-02
**/

class Matrix
{

public:

    int n, m; // dimensions
    int swaps = 0; // number of permutations during the elimination
    double **data; // dynamic array to store elements of matrix

    // constructor of class Matrix
    Matrix(int rows, int columns)
    {

        n = rows; // set number of rows
        m = columns; // set number of columns
        // allocating memory for array of arrays
        data = (double **) malloc(sizeof(double*) * n);
        for(int i = 0; i < n; i++)
            data[i] = (double*)malloc(sizeof(double) * m);

    }

    // overloading " >> " operator for class Matrix
    friend istream& operator >> (istream& in, const Matrix& matrix)
    {

        for(int i = 0; i < matrix.n; i++)
            for(int j = 0; j < matrix.m; j++)
                in >> matrix.data[i][j];
        return in;

    }

    // overloading " << " operator for class Matrix
    friend ostream& operator << (ostream& out, const Matrix& matrix)
    {

        for(int i = 0; i < matrix.n; i++)
        {
            for(int j = 0; j < matrix.m-1; j++)
                out << matrix.data[i][j] << " ";
            out << matrix.data[i][matrix.m-1] << "\n";
        }
        return out;

    }

    // overloading " = " operator for class Matrix
    Matrix& operator = (Matrix& other)
    {

        n = other.n; // set new
        m = other.m; // dimensions
        swaps = other.swaps; // transmit number of permutations
        data = other.data; // transfer elements
        return *this;

    }

    // overloading " + " operator for class Matrix
    Matrix& operator + (Matrix& other)
    {

        Matrix* matrixN = new Matrix(n, m); // creating instance of class Matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                matrixN -> data[i][j] = data[i][j] + other.data[i][j]; // store result of addition
        return *matrixN;

    }

    // overloading " - " operator for class Matrix
    Matrix& operator - (Matrix& other)
    {

        Matrix* matrixN = new Matrix(n, m); // creating instance of class Matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                matrixN -> data[i][j] = data[i][j] - other.data[i][j]; // store result of subtraction
        return *matrixN;

    }

    // overloading " * " operator for class Matrix
    Matrix& operator * (Matrix& other)
    {

        Matrix* product = new Matrix(n, other.m); // creating instance of class Matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j < other.m; j++)
                product -> data[i][j] = 0; // nullify all positions of new matrix
        for(int i = 0; i < n; i++)
            for(int j = 0; j< other.m; j++)
            {
                for(int k = 0; k < m; k++)
                    product -> data[i][j] += data[i][k] * other.data[k][j]; // store the obtained in multiplication values
                if(abs(product -> data[i][j]) < 0.01)
                    product -> data[i][j] = 0;
            }
        return *product;

    }

    // transposition method
    Matrix& transpose()
    {

        Matrix* matrixN = new Matrix(m, n); // creating instance of class Matrix
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                matrixN -> data[i][j] = data[j][i]; // store elements of rows in corresponding columns
        *this = *matrixN; // set obtained matrix instead of current
        return *matrixN;

    }

    // destructor of class
    ~Matrix()
    {
        for(int i = 0; i < n; i++)
            delete [] data[i];
        delete [] data;
    }

};

class SquareMatrix : public Matrix
{

public:
    SquareMatrix (int dimension):Matrix(dimension, dimension)
    {
        // creating new instance of class Matrix with received number of rows and columns
    }

};

class IdentityMatrix : public SquareMatrix
{
public:
    IdentityMatrix (int dimension):SquareMatrix(dimension)
    {
        for(int i = 0; i < dimension; i++)
            for(int j = 0; j < dimension; j++)
                i == j ? data[i][j] = 1 : data [i][j] = 0; // creating identity matrix, fill 0 in all positions except main diagonal
    }

};

class PermutationMatrix : public SquareMatrix
{

public:
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

class EliminationMatrix : public IdentityMatrix
{

public:
    EliminationMatrix (Matrix& matrix, int i1, int i2):IdentityMatrix(matrix.n)
    {
        i1--; // number of lines belongs
        i2--; // to [1; +inf]
        // check the potential division by 2
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

//used for determinant calculating
Matrix& LeadToUpperTriangularMatrix(Matrix &matrix)
{

    int step = 1, swaps = 0; // initialization number of steps and permutations
    Matrix *converted = new SquareMatrix(matrix.n); // creating new instance of class SquareMatrix
    *converted = matrix; // assign created matrix to received one
    for(int i = 0; i < converted -> n; i++)  // treat all rows of matrix
    {
        // find pivot with maximum absolute value
        // store its index in pivotIndex
        // store its value in pivotValue
        int pivotIndex = i;
        int pivotValue = abs(converted -> data[i][i]);
        for(int j = i; j < converted -> n; j++)
        {
            if (pivotValue < abs(converted -> data[j][i])) // find pivot with maximum absolute value
            {
                pivotIndex = j; // store index of found matrix
                pivotValue = abs(converted -> data[j][i]); // store value of found matrix
            }
        }
        // swap the current line with the found pivot line
        if(pivotIndex != i)
        {
            // create permutation matrix to swap the current line with the found pivot line
            cout << "step #" << step << ": permutation" << "\n";
            Matrix *P = new PermutationMatrix(converted -> n, pivotIndex + 1, i + 1); // P_{pivotline+1 i+1}
            *converted = *P * (*converted); // apply permutation matrix
            swaps++; // increment number of permutations
            cout << *converted; // print matrix after multiplying
        }
        for(int j = i + 1; j < converted -> n; j++)
        {
            // create elimination matrix to nullify all corresponding elements below the pivot
            cout << "step #" << step << ": elimination" << "\n";
            Matrix *E = new EliminationMatrix(*converted, j + 1, i + 1); // E_{j+1 i+1}
            *converted = *E * (*converted); // apply elimination matrix
            step ++; // increment number of steps
            cout << *converted; // print matrix after multiplying
        }
    }
    converted -> swaps = swaps; // assign number of permutations
    return *converted;

}

double CalculateDeterminant(Matrix &matrix)
{
    Matrix A = LeadToUpperTriangularMatrix(matrix);
    double determinant = 1;
    for (int i = 0; i < A.n; i++)
        determinant *= A.data[i][i]; // multiplying elements of the main diagonal
    A.swaps % 2 == 0 ? : determinant = -determinant; // considering number of permutations
    cout << "result:\n"; // print result
    cout << determinant;
    return determinant;

}

int main()
{
    cout.setf(ios::fixed); // set format of
    cout.precision(2);     // output values
    int aN; // dimension of matrix
    cin >> aN; // read dimension
    Matrix *A = new SquareMatrix(aN); // downcasting: creating instance of class SquareMatrix
    cin >> *A; // read matrix
    CalculateDeterminant(*A); // call function that calculates determinant
}