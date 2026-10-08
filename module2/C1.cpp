#include <iostream>
// int x = 5;
// int* p = new int(5);
// delete p;
// A pointer is a variable that holds the address
// syntax of pointers : 
//  & - "give me the address of.."
//  * - "give the value of this address"
// So how to use it properly?
int main(){
    int x = 10;
    int* p = &x;
    // print the address of the x by..
    // std::cout << p;
    // Now go to the address and give me the value..
    std::cout << *p;
    // Now write through the pointer..
    *p = 20; // pointer will assign the value 20 to the variable x
    std::cout << *p;
}