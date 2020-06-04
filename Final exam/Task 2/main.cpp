#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

/**
 * Danis Alukaev BS19-02
 * Final exam, part 2.
 * https://stepik.org/lesson/360846/step/7
 */

int main() {
    int l, r; // range of numbers
    cin >> l >> r; // input boundaries
    vector <int> result; // container of all elements that are satisfy mentioned in tack conditions
    for(int x = l; x <= r; x++){
        // for all numbers in given range
        vector <int> v; // container for digits of number
        int X = x; // copy the number to auxiliary variable
        while(X != 0){
            v.push_back(X % 10); // place digit in container
            X /= 10;
        }
        bool notFound = true;
        for(int i = 0; i < v.size() && notFound; i++)
            for(int j = i+1; j < v.size() && notFound; j++)
                if(v[i] == v[j])
                    // repeated digits found
                    notFound = false;
        if(notFound)
            // if repeated digits don't found
            result.push_back(x);
    }
    if(!result.empty())
        // if such numbers exist
        cout << result[0];
    else
        // otherwise
        cout << "-1";
}