#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

/**
 * Danis Alukaev BS19-02
 * Final exam, part 1.
 * https://stepik.org/lesson/360845/step/4
 */

int main() {
    int a; // number of elements in container
    vector<int> v; // container that stores elements
    cin >> a; // read the number of elements
    v.reserve(a); // set the number of elements
    for (int i = 0; i < a; i++) {
        int temp; // temporary variable to store element
        cin >> temp; // read element
        v.push_back(temp); // add element to a vector
    }
    bool terminate = false;
    for (int i = 0; i < v.size() && !terminate; i++) {
        for (int j = i + 1; j < v.size() && !terminate; j++) {
            // let us output the index of element that is square root firstly
            // as the order isn't mentioned in the task
            if (v[i] == v[j] * v[j]) {
                // if element in j-th place is square root of element in i-th place
                cout << j << '\n' << i;
                terminate = true; // terminate search
            }
            else if (v[j] == v[i] * v[i]) {
                // if element in i-th place is square root of element in j-th place
                cout << i << '\n' << j;
                terminate = true; // terminate search
            }
        }
    }
}