#include <iostream>

// void byValue (int x) { x = 100;} // copies x , caller's value unchanged
// void bypointer (int* x) { *x = 100;} // copies x , caller's value unchanged
// void byReference (int& x) { x = 100;} // copies x , caller's value unchanged

// const means it reads the pointer right to left
int main(){
    int x = 5;
    const int* p1 = &x; // in this line the " const int* " means the thing it points to is const
    int* const p2 = &x; // in this line the " int* const " means the pointer itself is a const
    std::cout << (*p2+1);
    std::cout << x;
    std::cout << (*p1+1);
    std::cout << x;
}