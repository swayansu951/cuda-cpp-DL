#include <iostream>

int main(){

    int arr[5] = {10,20,30,40,50};
    // assign a pointer as 'p'
    int* p = arr;
    std::cout << *p;
    std::cout << *(p+1); // this will print the next index value
    std::cout << p[2]; // this will print the 3rd index value

}